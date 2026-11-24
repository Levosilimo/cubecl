//! Wall-clock per call, batched so the queue is not the measurement. Compares the fused kernel against the
//! empty one, so the launch floor and the work are separable.
use cubecl::prelude::*;
use cubecl_environment::future::block_on;
use qwen35_norm::{nop, rmsnorm_residual_gate};

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
    let big: Vec<f32> = (0..d).map(|_| next()).collect();
    let w: Vec<f32> = (0..d).map(|_| next() * 0.1).collect();
    let a = client.create_from_slice(f32::as_bytes(&big));
    let b = client.create_from_slice(f32::as_bytes(&big));
    let c = client.create_from_slice(f32::as_bytes(&big));
    let wh = client.create_from_slice(f32::as_bytes(&w));
    let big24: Vec<f32> = (0..24 * d).map(|_| next()).collect();
    let out = client.empty(d * 4);
    let a24 = client.create_from_slice(f32::as_bytes(&big24));
    let b24 = client.create_from_slice(f32::as_bytes(&big24));
    let c24 = client.create_from_slice(f32::as_bytes(&big24));
    let out24 = client.empty(24 * d * 4);

    // BATCH=B launches per sync, so one sync amortises over B and the queue drains at device speed.
    for batch in [1usize, 16, 64] {
        let mut rows_out = Vec::new();
        for rows in [1usize, 24] {
            let tn = {
                let mut best = f64::MAX;
                for _ in 0..20 {
                    let t0 = std::time::Instant::now();
                    for _ in 0..batch {
                        nop::launch(&client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(32),
                            unsafe { BufferArg::from_raw_parts(out.clone(), d) });
                    }
                    block_on(client.sync()).unwrap();
                    let us = t0.elapsed().as_secs_f64() * 1e6 / batch as f64;
                    if us < best { best = us; }
                }
                best
            };
            let tf = {
                let mut best = f64::MAX;
                for _ in 0..20 {
                    let t0 = std::time::Instant::now();
                    for _ in 0..batch {
                        rmsnorm_residual_gate::launch::<f32, f32>(&client,
                            CubeCount::Static(rows as u32, 1, 1), CubeDim::new_1d((warps * 32) as u32),
                            unsafe { BufferArg::from_raw_parts(if rows == 1 { a.clone() } else { a24.clone() }, rows * d) },
                            unsafe { BufferArg::from_raw_parts(if rows == 1 { b.clone() } else { b24.clone() }, rows * d) },
                            unsafe { BufferArg::from_raw_parts(if rows == 1 { c.clone() } else { c24.clone() }, rows * d) },
                            unsafe { BufferArg::from_raw_parts(wh.clone(), d) },
                            unsafe { BufferArg::from_raw_parts(if rows == 1 { out.clone() } else { out24.clone() }, rows * d) },
                            1e-6f32, d, warps);
                    }
                    block_on(client.sync()).unwrap();
                    let us = t0.elapsed().as_secs_f64() * 1e6 / batch as f64;
                    if us < best { best = us; }
                }
                best
            };
            rows_out.push((rows, tn, tf));
        }
        for (rows, tn, tf) in rows_out {
            println!("  batch {batch:3}: rows {rows}  nop {tn:5.2} us   fused {tf:5.2} us   work {:.2} us", tf - tn);
        }
    }
}
