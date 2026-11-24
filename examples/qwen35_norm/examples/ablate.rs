//! Where does a fused kernel's time go? Launch floor, then the exp, then the reduction.
use cubecl::prelude::*;
use cubecl_environment::future::block_on;

use qwen35_norm::{nop, rmsnorm_residual_gate, rmsnorm_residual_nogate};

fn bench(mut f: impl FnMut(), n: usize) -> f64 {
    for _ in 0..10 { f(); }
    let mut best = f64::MAX;
    for _ in 0..30 {
        let t0 = std::time::Instant::now();
        for _ in 0..n { f(); }
        let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        if us < best { best = us; }
    }
    best
}

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
    let big: Vec<f32> = (0..2400 * d).map(|_| next()).collect();
    let w: Vec<f32> = (0..d).map(|_| next() * 0.1).collect();
    let a = client.create_from_slice(f32::as_bytes(&big));
    let b = client.create_from_slice(f32::as_bytes(&big));
    let c = client.create_from_slice(f32::as_bytes(&big));
    let wh = client.create_from_slice(f32::as_bytes(&w));
    let out = client.empty(2400 * d * 4);

    let _ = &client;
    let sync = || { let _ = client.read_one(out.clone()); };
    for rows in [1usize, 24] {
        let t_nop = bench(|| { nop::launch(&client, CubeCount::Static(1,1,1), CubeDim::new_1d(32), unsafe { BufferArg::from_raw_parts(out.clone(), 2400*d) }); }, 200);
        let t_gate = bench(|| { rmsnorm_residual_gate::launch::<f32,f32>(&client, CubeCount::Static(rows as u32,1,1), CubeDim::new_1d((warps*32) as u32),
            unsafe { BufferArg::from_raw_parts(a.clone(), rows*d) }, unsafe { BufferArg::from_raw_parts(b.clone(), rows*d) },
            unsafe { BufferArg::from_raw_parts(c.clone(), rows*d) }, unsafe { BufferArg::from_raw_parts(wh.clone(), d) },
            unsafe { BufferArg::from_raw_parts(out.clone(), rows*d) }, 1e-6f32, d, warps); }, 200);
        let t_nogate = bench(|| { rmsnorm_residual_nogate::launch::<f32,f32>(&client, CubeCount::Static(rows as u32,1,1), CubeDim::new_1d((warps*32) as u32),
            unsafe { BufferArg::from_raw_parts(a.clone(), rows*d) }, unsafe { BufferArg::from_raw_parts(b.clone(), rows*d) },
            unsafe { BufferArg::from_raw_parts(wh.clone(), d) },
            unsafe { BufferArg::from_raw_parts(out.clone(), rows*d) }, 1e-6f32, d, warps); }, 200);
        println!("  rows {rows}: nop {t_nop:.2} us   norm+residual {t_nogate:.2} us   +silu gate {t_gate:.2} us   (exp costs {:.2} us)", t_gate - t_nogate);
    }
}
