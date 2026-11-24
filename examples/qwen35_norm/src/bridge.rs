//! A C-ABI bridge so torch can call the fused kernel without a Python-side copy of the framework.
//!
//! THE DESIGN CONSTRAINT, measured before this was written: cubecl allocates with `cuMemAlloc` and exposes no
//! way to wrap a caller-owned device pointer into a `Handle`, so a torch tensor cannot be addressed zero-copy.
//! An upload costs 4.72 us against a 1.36-1.48 us launch (examples/ffi2.rs), so a per-call upload would erase
//! the fusion's benefit. Therefore:
//!
//!   * the Client is created ONCE and lives for the process;
//!   * handles are created ONCE per (buffer, shape) at `norm_init`, i.e. RESIDENT;
//!   * a call writes the caller's data into the resident handle, launches, and hands back a pointer to the
//!     resident output. No allocation on the hot path.
//!
//! The caller's data still has to cross, and that is the honest cost: `norm_upload` is d2d, so on a discrete
//! card it is a device-to-device copy (~10 us/MiB) and not a PCIe round trip. A future cubecl external-memory
//! API deletes it; nothing else in this design changes if it does.

use cubecl::prelude::*;
use cubecl_environment::future::block_on;
use cubecl_runtime::server::Handle;
use std::sync::OnceLock;
use std::sync::Mutex;

/// One resident buffer. `handle` is cubecl's allocation; `bytes` is its capacity.
struct Resident {
    handle: Handle,
    elems: usize,
}

struct Bridge {
    client: Client,
    x_norm: Option<Resident>,
    x_added: Option<Resident>,
    gate: Option<Resident>,
    weight: Option<Resident>,
    out: Option<Resident>,
    d: usize,
    warps: usize,
}

// The Client is not Sync, but all entry points here serialise on the Mutex, which is the contract this bridge
// offers: one kernel call at a time, which is what a decode step is anyway.
unsafe impl Send for Bridge {}

static BRIDGE: OnceLock<Mutex<Option<Bridge>>> = OnceLock::new();

fn bridge() -> &'static Mutex<Option<Bridge>> {
    BRIDGE.get_or_init(|| Mutex::new(None))
}

#[repr(C)]
pub struct NormDims {
    pub d: u32,
    pub warps: u32,
}

/// Create the resident buffers. Called once per model load, not per token.
///
/// # Safety
/// `d` must match the model's hidden size and the weights passed to later calls must have `d` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn norm_init(d: u32, warps: u32) -> i32 {
    let device = cubecl::Device::default();
    let client = device.client();
    let d = d as usize;
    let bytes = d * core::mem::size_of::<f32>();
    let mk = |n: usize| Resident {
        handle: client.empty(n * core::mem::size_of::<f32>()),
        elems: n,
    };
    let b = Bridge {
        x_norm: Some(mk(d)),
        x_added: Some(mk(d)),
        gate: Some(mk(d)),
        weight: Some(mk(d)),
        out: Some(mk(d)),
        d,
        warps: warps as usize,
        client,
    };
    let _ = bytes;
    *bridge().lock().unwrap() = Some(b);
    0
}

/// The device address of a resident buffer, so the caller can copy into it directly with its own stream.
/// `which`: 0 input to the norm, 1 residual, 2 gate, 3 weight, 4 output.
///
/// # Safety
/// `norm_init` must have been called and the returned handle must outlive its use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn norm_ptr(which: u32) -> u64 {
    let guard = bridge().lock().unwrap();
    let Some(b) = guard.as_ref() else {
        return 0;
    };
    let r = match which {
        0 => &b.x_norm,
        1 => &b.x_added,
        2 => &b.gate,
        3 => &b.weight,
        4 => &b.out,
        _ => return 0,
    };
    match r {
        Some(res) => {
            let addr = block_on(b.client.get_resource_addr(&res.handle));
            addr.unwrap_or(0)
        }
        None => 0,
    }
}

/// Run the fused kernel over `rows` rows held in the resident buffers.
///
/// The caller is expected to have filled `x_norm`, `x_added`, `gate` and `weight` with host-to-device copies into
/// the pointers returned by `norm_ptr`. `out` is written in place and read back through the same pointer.
///
/// # Safety
/// `norm_init` must have been called, and all pointers returned by `norm_ptr` must still be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn norm_run(rows: u32, eps: f32) -> i32 {
    let guard = bridge().lock().unwrap();
    let Some(b) = guard.as_ref() else {
        return -1;
    };
    let d = b.d;
    let (Some(xn), Some(xa), Some(g), Some(w), Some(o)) =
        (&b.x_norm, &b.x_added, &b.gate, &b.weight, &b.out)
    else {
        return -2;
    };
    let n = rows as usize * d;
    if xn.elems < n || xa.elems < n || g.elems < n || o.elems < n {
        return -3;
    }
    crate::rmsnorm_residual_gate::launch::<f32, f32>(
        &b.client,
        CubeCount::Static(rows, 1, 1),
        CubeDim::new_1d((b.warps * 32) as u32),
        BufferArg::from_raw_parts(xn.handle.clone(), n),
        BufferArg::from_raw_parts(xa.handle.clone(), n),
        BufferArg::from_raw_parts(g.handle.clone(), n),
        BufferArg::from_raw_parts(w.handle.clone(), d),
        BufferArg::from_raw_parts(o.handle.clone(), n),
        eps,
        d,
        b.warps,
    );
    match block_on(b.client.sync()) {
        Ok(()) => 0,
        Err(_) => -4,
    }
}
