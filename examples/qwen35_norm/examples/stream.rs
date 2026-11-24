//! THE question that decides whether a step entry can interleave with our existing int4 GEMV kernels:
//! which CUDA stream does cubecl launch on, and does it respect an externally-set current stream?
use cubecl::prelude::*;

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    println!("  cubecl stream id: {:?}", client.stream_id());
    // if cubecl exposed a way to select the stream, a torch-captured graph could contain its launches
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../crates/cubecl-runtime/src/client.rs")).unwrap();
    let n = src.lines().filter(|l| l.contains("pub fn") && l.contains("stream")).count();
    println!("  public stream-related methods on Client: {n}");
    for l in src.lines().filter(|l| l.contains("pub fn") && l.contains("stream")) {
        println!("    {}", l.trim());
    }
}
