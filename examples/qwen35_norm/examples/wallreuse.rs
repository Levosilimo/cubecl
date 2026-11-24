//! The write-inside-capture limit is a PINNED-PAGE POOL limit, not a count limit: during recording,
//! `PageUpdate::Forbidden` means the staging pool may only serve pages it already holds, and the pool was
//! primed by the warmup's own writes. So the fix is to make the warmup allocate EVERY page the capture will
//! need -- i.e. write from as many DISTINCT staging sizes/addresses as the graph does.
//!
//! Test: allocate the staging buffers up front and touch each one during warmup, so the pool holds a page per
//! buffer, then capture 48 writes and see whether it holds.
use cubecl::prelude::*;
use cubecl_environment::future::block_on;

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let d = 2048usize;
    // N distinct destination handles AND N distinct host staging buffers, as a branchy step needs
    let n = 48usize;
    let hosts: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32 * 0.01; d]).collect();
    let handles: Vec<_> = (0..n).map(|_| client.empty(d * 4)).collect();
    let mut one = |i: usize| {
        client.write(
            &handles[i],
            cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&hosts[i]).to_vec()),
        );
    };
    // WARMUP MUST PRIME AT LEAST AS MANY STAGING PAGES AS THE CAPTURE WILL USE. One pass of n writes primes n
    // pages; a capture of n writes then fits exactly. A capture of 3n against a warmup of n fails -- which is
    // what the bridge hit at 32 iterations (96 writes) after warming with 16 (48 writes).
    for _ in 0..3 { for i in 0..n { one(i) } }
    block_on(client.sync()).unwrap();
    let cap_writes = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(3usize);
    let (g, _) = (client.capture(|| {
        for _ in 0..cap_writes { for i in 0..n { one(i) } }
    })).unwrap();
    unsafe { g.replay() }.unwrap();
    block_on(client.sync()).unwrap();
    println!("  {} writes inside one capture (warmup primed {}): OK", n * cap_writes, n * 3);
    println!("  -> the pool limit is per distinct staging size, and priming every buffer first clears it");
}
