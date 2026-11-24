//! The plain norm in pure Rust: f16 in, f16 out, f32 accumulate. Isolates the kernel from the bridge (and from
//! cubecl's own stream choice, which is PerThread and so NOT torch's current stream outside a capture).
use cubecl::prelude::*;
use cubecl_environment::future::block_on;
use half::f16;
use qwen35_norm::rmsnorm_plain;

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let d = 2048usize;
    let warps = 16usize;
    let x: Vec<f16> = (0..d).map(|i| f16::from_f32(((i as f32 * 0.37) % 3.0) - 1.5)).collect();
    let w: Vec<f16> = (0..d).map(|i| f16::from_f32((i as f32 * 0.01) - 10.0)).collect();
    let xh = client.create_from_slice(u16::as_bytes(&x.iter().map(|v| v.to_bits()).collect::<Vec<_>>()));
    let wh = client.create_from_slice(u16::as_bytes(&w.iter().map(|v| v.to_bits()).collect::<Vec<_>>()));
    let out = client.empty(d * 2);
    rmsnorm_plain::launch::<f16, f16>(
        &client, CubeCount::Static(1, 1, 1), CubeDim::new_1d((warps * 32) as u32),
        unsafe { BufferArg::from_raw_parts(xh, d) },
        unsafe { BufferArg::from_raw_parts(wh, d) },
        unsafe { BufferArg::from_raw_parts(out.clone(), d) },
        1e-6f32, d, warps,
    );
    block_on(client.sync()).unwrap();
    let bytes = client.read_one(out).unwrap();
    let got: Vec<f16> = u16::from_bytes(&bytes).iter().map(|b| f16::from_bits(*b)).collect();
    // reference in f32
    let mut sq = 0.0f32;
    for v in &x { let f = v.to_f32(); sq += f * f; }
    let rstd = 1.0f32 / (sq / d as f32 + 1e-6).sqrt();
    let (mut worst, mut wsum) = (0.0f32, 0.0f32);
    for i in 0..d {
        let want = x[i].to_f32() * rstd * (1.0 + w[i].to_f32());
        worst = worst.max((got[i].to_f32() - want).abs());
        wsum += want.abs();
    }
    println!("  f16 out: worst abs err {worst:.3e}  mean |want| {:.3}", wsum / d as f32);
    println!("  got[0..4] {:?}", &got[0..4].iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    println!("  first want {:.4}", x[0].to_f32() * rstd * (1.0 + w[0].to_f32()));
}
