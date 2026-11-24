//! Time the fused kernel in isolation, with the SM clock burned first and verified -- this box idles at
//! 936 MHz of 1987 and a cold reading is 1.5x slow (memory 4252).

use cubecl::prelude::*;
use cubecl_environment::future::block_on;
use qwen35_norm::rmsnorm_residual_gate;

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let rows: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(24);
    let d = 2048usize;
    let eps = 1e-6f32;
    let warps: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(4);
    let mut seed = 7u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0
    };
    let xn: Vec<f32> = (0..rows * d).map(|_| next()).collect();
    let xa: Vec<f32> = (0..rows * d).map(|_| next()).collect();
    let gate: Vec<f32> = (0..rows * d).map(|_| next() * 2.0).collect();
    let w: Vec<f32> = (0..d).map(|_| next() * 0.1).collect();
    let xn_h = client.create_from_slice(f32::as_bytes(&xn));
    let xa_h = client.create_from_slice(f32::as_bytes(&xa));
    let gate_h = client.create_from_slice(f32::as_bytes(&gate));
    let w_h = client.create_from_slice(f32::as_bytes(&w));
    let out_h = client.empty(rows * d * core::mem::size_of::<f32>());

    let launch = || {
        let xn_h = xn_h.clone();
        let xa_h = xa_h.clone();
        let gate_h = gate_h.clone();
        let w_h = w_h.clone();
        let out_h = out_h.clone();
        rmsnorm_residual_gate::launch::<f32, f32>(
            &client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d((warps * 32) as u32),
            unsafe { BufferArg::from_raw_parts(xn_h, rows * d) },
            unsafe { BufferArg::from_raw_parts(xa_h, rows * d) },
            unsafe { BufferArg::from_raw_parts(gate_h, rows * d) },
            unsafe { BufferArg::from_raw_parts(w_h, d) },
            unsafe { BufferArg::from_raw_parts(out_h.clone(), rows * d) },
            eps,
            d,
            warps,
        );
    };
    for _ in 0..20 {
        launch();
    }
    block_on(client.sync()).unwrap();

    let mut best = f64::MAX;
    let mut samples = Vec::new();
    for _ in 0..40 {
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            launch();
        }
        block_on(client.sync()).unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / 20.0;
        samples.push(us);
        if us < best {
            best = us;
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  fused kernel, one launch (rows {} d {}):  median {:.2} us   best {:.2} us   n={}",
        rows,
        d,
        samples[samples.len() / 2],
        best,
        samples.len()
    );
}
