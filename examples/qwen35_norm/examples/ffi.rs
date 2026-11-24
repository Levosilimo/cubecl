//! The bridge shape: a C-ABI entry point that takes raw device pointers, so a torch tensor can be handed to a
//! cubecl kernel without a Python-side copy of the framework. Two handles are created per buffer per call --
//! measured here to confirm that is not the new bottleneck (torch is 1.36-1.48 us/launch captured, so the
//! bridge must stay well under that).

use cubecl::prelude::*;
use cubecl_environment::future::block_on;
use cubecl_runtime::server::Handle;
use qwen35_norm::rmsnorm_residual_gate;

/// Everything the kernel needs, passed across the FFI boundary as one struct so the signature can grow without
/// breaking the caller.
#[repr(C)]
pub struct NormArgs {
    pub x_norm: u64,   // device pointer
    pub x_added: u64,
    pub gate: u64,
    pub weight: u64,
    pub out: u64,
    pub rows: u32,
    pub d: u32,
    pub warps: u32,
    pub eps: f32,
}

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let _ = &client;
    println!("  bridge: cubecl's client is process-wide per device, so a Rust cdylib can hold one and expose");
    println!("  C functions that take raw pointers. The cost to measure is HANDLE CREATION per call:");
    // how long does it take to make a Handle from an existing allocation? `create_from_slice` copies, so the
    // real bridge needs a handle over caller-owned memory, which cubecl does not expose -- measure the copy.
    let d = 2048usize;
    let buf = vec![0.5f32; d * 4];
    let mut best = f64::MAX;
    for _ in 0..200 {
        let t = std::time::Instant::now();
        let _h: Handle = client.create_from_slice(f32::as_bytes(&buf));
        block_on(client.sync()).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e6);
    }
    println!("  create_from_slice + sync, {} KiB: {:.2} us", d * 4 * 4 / 1024, best);
    println!("  -> a per-call copy at this cost would dominate a 1.4 us launch, so the bridge must either");
    println!("     keep the activations resident in cubecl handles across the step, or accept an upload");
    println!("     only where a fused kernel span replaces >=8 torch kernels.");
}
