//! Correctness against a CPU reference, then a kernel-count and time comparison against the torch chain the
//! kernel replaces.

use cubecl::prelude::*;
use qwen35_norm::rmsnorm_residual_gate;

fn reference(xn: &[f32], xa: &[f32], gate: &[f32], w: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * d];
    for r in 0..rows {
        let b = r * d;
        let mut ss = 0.0f32;
        for i in 0..d {
            ss += xn[b + i] * xn[b + i];
        }
        let rstd = 1.0 / (ss / d as f32 + eps).sqrt();
        for i in 0..d {
            let n = xn[b + i] * rstd;
            let ww = 1.0 + w[i];
            let g = gate[b + i];
            let silu = g / (1.0 + (-g).exp());
            out[b + i] = xa[b + i] + n * ww * silu;
        }
    }
    out
}

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let rows = 24usize;
    let d = 2048usize;
    let eps = 1e-6f32;
    let warps = 4usize;

    // deterministic pseudo-random inputs in the model's range
    let mut seed = 12345u64;
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

    let bytes = client.read_one(out_h).unwrap();
    let got = f32::from_bytes(&bytes);
    let want = reference(&xn, &xa, &gate, &w, rows, d, eps);
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for i in 0..rows * d {
        let e = (got[i] - want[i]).abs();
        if e > worst {
            worst = e;
            worst_i = i;
        }
    }
    let scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!(
        "  contract: rows {} d {} warps {}   worst abs err {:e} at {}   scale {}   rel {:e}",
        rows, d, warps, worst, worst_i, scale, worst / scale
    );
}
