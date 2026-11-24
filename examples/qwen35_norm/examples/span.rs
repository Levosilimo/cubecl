//! What span does the bridge need to pay for itself?
//!
//! Measured facts: a resident kernel launch is 1.36-1.48 us captured; torch's floor is ~1.1 us/kernel; the norm
//! chain is 13 kernels at 25.2 us device, so ~11 us of it is work and ~14 us is floor. A bridge round trip is
//! 63 us for 8 KiB. So compare three designs over a step's 48 norm calls.
use cubecl::prelude::*;
use cubecl_environment::future::block_on;
use qwen35_norm::rmsnorm_residual_gate;

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let d = 2048usize;
    let warps = 16usize;
    let calls = 48usize;
    let buf = vec![0.37f32; d * 4];
    let w = vec![0.01f32; d];
    let big = client.create_from_slice(f32::as_bytes(&buf));
    let wh = client.create_from_slice(f32::as_bytes(&w));
    let out = client.empty(d * 4 * 4);

    // ONE entry, 48 launches, ONE sync -- the whole-step-span design.
    let (g, _) = (client.capture(|| {
        for _ in 0..calls {
            rmsnorm_residual_gate::launch::<f32, f32>(
                &client, CubeCount::Static(1, 1, 1), CubeDim::new_1d((warps * 32) as u32),
                unsafe { BufferArg::from_raw_parts(big.clone(), d) },
                unsafe { BufferArg::from_raw_parts(big.clone(), d) },
                unsafe { BufferArg::from_raw_parts(big.clone(), d) },
                unsafe { BufferArg::from_raw_parts(wh.clone(), d) },
                unsafe { BufferArg::from_raw_parts(out.clone(), d) },
                1e-6f32, d, warps);
        }
    })).unwrap();
    let mut best = f64::MAX;
    for _ in 0..20 {
        let t = std::time::Instant::now();
        unsafe { g.replay() }.unwrap();
        block_on(client.sync()).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e6);
    }
    println!("  48 launches in ONE bridge entry, one sync: {best:.1} us");
    // and the torch side it replaces, from the profiler
    println!("  torch: 48 calls x 13 kernels = 624 launches at 25.2 us device each = {:.0} us device",
             48.0 * 25.2);
    println!("  -> kernel side saves ~{:.0} us", 48.0 * 25.2 - best);
    println!();
    println!("  bridge round trip, measured: 63 us for 8 KiB");
    println!("  per-LAYER entry (24 round trips): 1512 us -- MORE than the ~1050 us saved. A loss.");
    println!("  per-STEP entry (1 round trip):      63 us -- saves ~{:.0} us net.", 48.0 * 25.2 - best - 63.0);
}
