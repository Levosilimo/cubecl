//! Is the 59 us the upload or the sync? And does a resident handle make a per-launch cost disappear?
use cubecl::prelude::*;
use cubecl_environment::future::block_on;
use qwen35_norm::rmsnorm_residual_gate;

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let d = 2048usize;
    let warps = 16usize;
    let buf = vec![0.5f32; d];
    let a = client.create_from_slice(f32::as_bytes(&buf));
    let b = client.create_from_slice(f32::as_bytes(&buf));
    let c = client.create_from_slice(f32::as_bytes(&buf));
    let w = client.create_from_slice(f32::as_bytes(&buf));
    let out = client.empty(d * 4);

    // 1. upload alone, no sync
    let mut up = f64::MAX;
    for _ in 0..50 {
        let t = std::time::Instant::now();
        for _ in 0..50 { let _ = client.create_from_slice(f32::as_bytes(&buf)); }
        up = up.min(t.elapsed().as_secs_f64() * 1e6 / 50.0);
    }
    // 2. upload + sync
    let mut ups = f64::MAX;
    for _ in 0..50 {
        let t = std::time::Instant::now();
        for _ in 0..50 { let _ = client.create_from_slice(f32::as_bytes(&buf)); }
        block_on(client.sync()).unwrap();
        ups = ups.min(t.elapsed().as_secs_f64() * 1e6 / 50.0);
    }
    // 3. resident handles: kernel only
    let fire = || {
        rmsnorm_residual_gate::launch::<f32, f32>(&client,
            CubeCount::Static(1, 1, 1), CubeDim::new_1d((warps * 32) as u32),
            unsafe { BufferArg::from_raw_parts(a.clone(), d) },
            unsafe { BufferArg::from_raw_parts(b.clone(), d) },
            unsafe { BufferArg::from_raw_parts(c.clone(), d) },
            unsafe { BufferArg::from_raw_parts(w.clone(), d) },
            unsafe { BufferArg::from_raw_parts(out.clone(), d) },
            1e-6f32, d, warps);
    };
    for _ in 0..20 { fire(); }
    block_on(client.sync()).unwrap();
    let (g, _) = (client.capture(|| { for _ in 0..40 { fire(); } })).unwrap();
    let mut cap = f64::MAX;
    for _ in 0..20 {
        let t = std::time::Instant::now();
        unsafe { g.replay() }.unwrap();
        block_on(client.sync()).unwrap();
        cap = cap.min(t.elapsed().as_secs_f64() * 1e6 / 40.0);
    }
    println!("  upload only            {up:6.2} us");
    println!("  upload + sync          {ups:6.2} us");
    println!("  kernel, resident+graph {cap:6.2} us");
    // 32 KiB is L2-sized; the real bridge would keep activations resident, so 3 is the number that matters

}
