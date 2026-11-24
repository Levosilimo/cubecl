//! Does cubecl's captured launch cost scale with the number of buffer arguments?
use cubecl::prelude::*;
use cubecl_environment::future::block_on;
use qwen35_norm::{said1, said5};

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let n_el = 2048usize;
    let buf: Vec<f32> = (0..n_el).map(|i| i as f32 * 0.001).collect();
    let a = client.create_from_slice(f32::as_bytes(&buf));
    let b = client.create_from_slice(f32::as_bytes(&buf));
    let c = client.create_from_slice(f32::as_bytes(&buf));
    let d = client.create_from_slice(f32::as_bytes(&buf));
    let out = client.empty(n_el * 4);
    let grid = CubeCount::Static(1, 1, 1);
    let dim = CubeDim::new_1d(256);
    for n in [40usize, 400] {
        let (g1, _) = (client.capture(|| {
            for _ in 0..n { said1::launch(&client, grid.clone(), dim.clone(),
                unsafe { BufferArg::from_raw_parts(a.clone(), n_el) },
                unsafe { BufferArg::from_raw_parts(out.clone(), n_el) }); }
        })).unwrap();
        let (g5, _) = (client.capture(|| {
            for _ in 0..n { said5::launch(&client, grid.clone(), dim.clone(),
                unsafe { BufferArg::from_raw_parts(a.clone(), n_el) },
                unsafe { BufferArg::from_raw_parts(b.clone(), n_el) },
                unsafe { BufferArg::from_raw_parts(c.clone(), n_el) },
                unsafe { BufferArg::from_raw_parts(d.clone(), n_el) },
                unsafe { BufferArg::from_raw_parts(out.clone(), n_el) }); }
        })).unwrap();
        let mut t1 = f64::MAX;
        let mut t5 = f64::MAX;
        for _ in 0..12 {
            let t = std::time::Instant::now();
            unsafe { g1.replay() }.unwrap(); block_on(client.sync()).unwrap();
            t1 = t1.min(t.elapsed().as_secs_f64() * 1e6 / n as f64);
            let t = std::time::Instant::now();
            unsafe { g5.replay() }.unwrap(); block_on(client.sync()).unwrap();
            t5 = t5.min(t.elapsed().as_secs_f64() * 1e6 / n as f64);
        }
        println!("  n={n:4}: 2-arg kernel {t1:.2} us/launch   5-arg kernel {t5:.2} us/launch   delta {:.2}",
                 t5 - t1);
    }
}
