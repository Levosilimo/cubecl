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
use cubecl::cuda::compute::CudaServer;
use cubecl::cuda::compute::storage::gpu::GpuResource;
use cubecl_environment::future::block_on;
use cubecl_runtime::server::Handle;
use cubecl_runtime::storage::ManagedResource;
use std::sync::{Mutex, OnceLock};

/// One resident buffer. `handle` is cubecl's allocation, `bound` keeps it in place and carries its device
/// address, and `elems` is its capacity.
///
/// Holding the binding is what makes the address safe to hand out: a bound allocation is never freed, handed
/// out again or relocated, so `address()` stays valid for as long as this struct lives.
struct Resident {
    handle: Handle,
    bound: Option<ManagedResource<GpuResource>>,
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
pub unsafe extern "C" fn norm_init(d: u32, warps: u32, max_rows: u32) -> i32 {
    let device = cubecl::Device::default();
    let client = device.client();
    let d = d as usize;
    // buffers are sized for the LARGEST span once, so a call never reallocates: re-allocation on the hot path is
    // what the whole residency design exists to avoid.
    let n = d * max_rows.max(1) as usize;
    let mk = |n: usize| {
        let handle = client.empty(n * core::mem::size_of::<f32>());
        // `get_resource` binds the allocation for as long as the result is held, which is exactly what makes
        // `address()` below valid to hand to another framework.
        let bound = client.get_resource::<CudaServer>(handle.clone()).ok();
        Resident { handle, bound, elems: n }
    };
    let b = Bridge {
        x_norm: Some(mk(n)),
        x_added: Some(mk(n)),
        gate: Some(mk(n)),
        weight: Some(mk(d)),
        out: Some(mk(n)),
        d,
        warps: warps as usize,
        client,
    };
    *bridge().lock().unwrap() = Some(b);
    0
}

/// The device address of a resident buffer, or 0 if it has none. `which` is 0 input, 1 residual, 2 gate,
/// 3 weight, 4 output.
///
/// This is the piece that removes the boundary copy: with a real device pointer the caller can `memcpy` d2d
/// (or hand it to DLPack) instead of staging through the host, and a d2d copy inside a capture costs ~1.4 us
/// instead of the 63 us host round trip.
///
/// # Safety
/// `norm_init` must have run, and the pointer is valid only while the bridge lives -- i.e. until `norm_init`
/// is called again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn norm_ptr(which: u32) -> u64 {
    let guard = bridge().lock().unwrap();
    let Some(b) = guard.as_ref() else { return 0 };
    let r = match which {
        0 => &b.x_norm,
        1 => &b.x_added,
        2 => &b.gate,
        3 => &b.weight,
        4 => &b.out,
        _ => return 0,
    };
    match r.as_ref().and_then(|res| res.bound.as_ref()) {
        Some(m) => m.resource().ptr,
        None => 0,
    }
}

/// Copy the caller's bytes into resident buffer `which`. `ptr` MUST BE A HOST POINTER holding `bytes` bytes.
///
/// A device pointer would be dereferenced by the host memcpy below and segfault -- the first version of this
/// did exactly that with a torch `data_ptr()`, which is why the ABI takes host data and the caller stages it.
/// That staging is the 33 us per 8 KiB round trip the bridge costs and the reason a shipping design is one
/// `step()` entry rather than a per-sublayer call.
///
/// The copy is a HOST round trip because cubecl exposes no device address for a local buffer (`address()` exists
/// only on `PeerBinding`). Measured at 33 us per 8 KiB, so this entry point exists to VALIDATE the kernel inside
/// the model, not to ship: the shipping shape is one `step()` entry per decode step (windows/analysis/cubecl_spike.md).
///
/// # Safety
/// `ptr` must point to `bytes` readable HOST bytes, and `norm_init` must have run.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn norm_write(which: u32, ptr: *const u8, bytes: u64) -> i32 {
    let guard = bridge().lock().unwrap();
    let Some(b) = guard.as_ref() else { return -1 };
    let r = match which {
        0 => &b.x_norm,
        1 => &b.x_added,
        2 => &b.gate,
        3 => &b.weight,
        _ => return -2,
    };
    let Some(res) = r else { return -2 };
    if bytes as usize > res.elems * 4 {
        return -3;
    }
    let slice = core::slice::from_raw_parts(ptr, bytes as usize);
    b.client.write(&res.handle, cubecl::bytes::Bytes::from_bytes_vec(slice.to_vec()));
    // NO SYNC, for the same reason as `norm_run`: a sync inside a capture invalidates it. The write is
    // enqueued on the current stream, which is the capturing stream, so the graph orders it.
    if SYNC_WRITES.load(core::sync::atomic::Ordering::Relaxed) {
        block_on(b.client.sync()).map(|_| 0).unwrap_or(-4)
    } else {
        0
    }
}

/// Whether `norm_write` should wait for its copy. Off by default so a captured call is legal; `norm_sync` is the
/// explicit wait for callers outside a graph.
static SYNC_WRITES: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Turn the synchronous write on or off. Off is the captured-graph mode.
///
/// # Safety
/// Must be called from the same thread that performs the writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn norm_set_sync_writes(on: u32) {
    SYNC_WRITES.store(on != 0, core::sync::atomic::Ordering::Relaxed);
}

/// Copy the resident output back to the caller.
///
/// # Safety
/// `ptr` must point to `bytes` writable HOST bytes, and `norm_init` must have run. Two copies happen: the read
/// lands in a host staging buffer and then is memcpy'd out, which is what makes a read 30 us.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn norm_read_out(ptr: *mut u8, bytes: u64) -> i32 {
    let guard = bridge().lock().unwrap();
    let Some(b) = guard.as_ref() else { return -1 };
    let Some(res) = &b.out else { return -2 };
    if bytes as usize > res.elems * 4 {
        return -3;
    }
    let Ok(data) = b.client.read_one(res.handle.clone()) else { return -4 };
    let dst = core::slice::from_raw_parts_mut(ptr, bytes as usize);
    dst.copy_from_slice(&data[..bytes as usize]);
    0
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
    // NO SYNC. A sync inside a capture invalidates it (`cudaErrorStreamCaptureInvalidated`), and cubecl launches
    // on `StreamId::current()` -- the current torch stream -- so an unsynced launch is ordered by the graph
    // exactly like any other kernel in it. This is the entry point that costs NOTHING per call: no round trip,
    // no sync, no bridge. The caller reads the result with torch once the graph has replayed.
    let _ = &b.client;
    0
}

/// Host-side wait, for callers outside a graph. Deliberately separate from `norm_run` so a captured call is
/// never accidentally synchronised.
///
/// # Safety
/// `norm_init` must have run.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn norm_sync() -> i32 {
    let guard = bridge().lock().unwrap();
    let Some(b) = guard.as_ref() else { return -1 };
    match block_on(b.client.sync()) {
        Ok(()) => 0,
        Err(_) => -4,
    }
}
