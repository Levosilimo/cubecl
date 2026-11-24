//! THE decisive number for a fused forward: cubecl's per-launch floor is ~6.8 us on this box against the card's
//! own 1.13 us. A 40-launch fused forward would pay 272 us of marshalling unless the launches are captured into
//! one CUDA graph. So capture N of them and measure replay.

use cubecl::prelude::*;
use cubecl_environment::future::block_on;
use cubecl_runtime::server::Handle;
use qwen35_norm::rmsnorm_residual_gate;

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let d = 2048usize;
    let warps = 16usize;
    let mut seed = 7u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0
    };
    let buf: Vec<f32> = (0..d).map(|_| next()).collect();
    let w: Vec<f32> = (0..d).map(|_| next() * 0.1).collect();
    let a = client.create_from_slice(f32::as_bytes(&buf));
    let b = client.create_from_slice(f32::as_bytes(&buf));
    let c = client.create_from_slice(f32::as_bytes(&buf));
    let wh = client.create_from_slice(f32::as_bytes(&w));
    let out = client.empty(d * 4);

    // ONE launch, in a closure returning its own handles so the graph can own them.
    let fire = |a: Handle, b: Handle, c: Handle, w: Handle, o: Handle| {
        rmsnorm_residual_gate::launch::<f32, f32>(
            &client, CubeCount::Static(1, 1, 1), CubeDim::new_1d((warps * 32) as u32),
            unsafe { BufferArg::from_raw_parts(a, d) },
            unsafe { BufferArg::from_raw_parts(b, d) },
            unsafe { BufferArg::from_raw_parts(c, d) },
            unsafe { BufferArg::from_raw_parts(w, d) },
            unsafe { BufferArg::from_raw_parts(o, d) },
            1e-6f32, d, warps);
    };
    for n in [8usize, 20, 40, 200, 1000] {
        // eager: n launches per sync
        let mut eager = f64::MAX;
        for _ in 0..15 {
            let t0 = std::time::Instant::now();
            for _ in 0..n { fire(a.clone(), b.clone(), c.clone(), wh.clone(), out.clone()); }
            block_on(client.sync()).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6;
            if us < eager { eager = us; }
        }
        // captured: n launches in one graph, replay timed
        let (graph, _) = (client.capture(|| {
            for _ in 0..n { fire(a.clone(), b.clone(), c.clone(), wh.clone(), out.clone()); }
        })).unwrap();
        let mut cap = f64::MAX;
        for _ in 0..15 {
            let t0 = std::time::Instant::now();
            unsafe { graph.replay() }.unwrap();
            block_on(client.sync()).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6;
            if us < cap { cap = us; }
        }
        println!("  n={n:3}: eager {eager:7.1} us total ({:.2} us/launch)   captured {cap:7.1} us total ({:.2} us/launch)   {:.1}x",
                 eager / n as f64, cap / n as f64, eager / cap);
    }
}
