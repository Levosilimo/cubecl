//! Does `client.write` work on a handle the bridge created, in pure Rust with no FFI?
use cubecl::prelude::*;
use cubecl_environment::future::block_on;

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let d = 2048usize;
    let h = client.empty(d * 4);
    let data = vec![0.5f32; d];
    println!("  writing {} bytes", d * 4);
    client.write(&h, cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&data).to_vec()));
    println!("  wrote, syncing");
    block_on(client.sync()).unwrap();
    println!("  synced");
    let back = client.read_one(h).unwrap();
    println!("  read back {} bytes, first {}", back.len(), f32::from_bytes(&back)[0]);
}
