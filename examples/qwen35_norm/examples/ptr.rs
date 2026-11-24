//! There is NO public accessor for a local buffer's device address -- `address()` exists only on `PeerBinding`
//! (cross-device sharing), not on a `BufferBinding`. So the bridge cannot hand torch a pointer to write into.
//! That leaves `client.write(handle, Bytes)` / `read_one(handle)`, i.e. a host round trip. Price it at the two
//! sizes that matter: one layer's activations, and a whole step's.
use cubecl::prelude::*;
use cubecl_environment::future::block_on;

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    // 4 KiB = one [1,1,2048] fp16 tensor... in f32 terms 8 KiB. Measure 8 KiB and 768 KiB.
    for kib in [8usize, 768] {
        let n = kib * 1024 / 4;
        let h = client.create_from_slice(f32::as_bytes(&vec![0.5f32; n]));
        let data = vec![0.5f32; n];
        let mut w = f64::MAX;
        let mut r = f64::MAX;
        for _ in 0..40 {
            let t = std::time::Instant::now();
            client.write(&h, cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&data).to_vec()));
            block_on(client.sync()).unwrap();
            w = w.min(t.elapsed().as_secs_f64() * 1e6);
            let t = std::time::Instant::now();
            let _ = client.read_one(h.clone()).unwrap();
            r = r.min(t.elapsed().as_secs_f64() * 1e6);
        }
        println!("  {kib:4} KiB: write+sync {w:6.2} us   read {r:6.2} us   round trip {:.2} us", w + r);
    }
    println!();
    println!("  a decoder layer's activations are ~8 KiB (hidden_states in and out);");
    println!("  24 layers x 2 round trips is the per-step bridge cost if the span is one layer.");
}
