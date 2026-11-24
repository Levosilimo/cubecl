//! The RMSNorm + residual + gate-mul sublayer of a Qwen3.5 decode step, written as a fused `#[cube]` kernel.
//!
//! Why this slice: the census of a captured decode step puts 1115 non-matmul kernels in it at 3.40 ms, of
//! which 1.26 ms is the empty-launch floor and 2.14 ms is work on tensors of one to a few thousand elements.
//! The op table says the same thing from the CPU side -- 496 `view`, 494 `empty`, 470 `reshape`, 464 `to`,
//! 252 `_to_copy`, and a `mean`/`rsqrt`/`pow` triple 61 times over. Those 183 ops are one RMSNorm per norm per
//! layer, each emitting several kernels.
//!
//! This kernel does what `Qwen3_5RMSNorm` + the residual add + the gate multiply do, in ONE launch:
//!
//!     y = x_added + (norm(x_norm) * (1 + w)) * silu(z)
//!
//! The reference does it as, per call: `x.float()`, `pow(2)`, `mean(-1)`, `rsqrt`, `mul`, `mul`, `to(dtype)`,
//! `add`, `mul`, `silu` -- 10 ops and roughly 7 kernels, over 48 call sites per step. One launch each.
//!
//! Numerics follow the reference exactly, because a fused forward is worthless if it changes the answer: the
//! sum of squares accumulates in `f32`, the normalisation multiplies in `f32`, the weight is `1.0 + w` (Qwen3.5
//! stores a zero-initialised delta, unlike Llama), and the single rounding to the model dtype happens once, at
//! the output.

use cubecl::prelude::*;

/// One block per row. `D` is the row width (hidden size), comptime so the reductions unroll.
///
/// `plane_sum` reduces within a warp and `sync_cube` plus shared memory combines warps, which is the shape
/// `block_sum` in `csrc/kernels/gdn_layer.cu` needed three attempts to get right -- a partial-width warp
/// reduction with a shared staging array and a trailing barrier after the broadcast.
#[cube(launch)]
pub fn rmsnorm_residual_gate<F: Float, E: Float>(
    x_norm: &[F],     // [rows, D] input to the norm
    x_added: &[F],   // [rows, D] the residual it is added to
    gate: &[F],      // [rows, D] the silu gate (pass x_added itself and a zero weight for plain norm+residual)
    weight: &[E],    // [D] the norm weight, stored as a zero-centred delta
    out: &mut [F],   // [rows, D]
    eps: f32,
    #[comptime] d: usize,
    #[comptime] warps: usize,
) {
    let plane = 32u32;
    let lane = UNIT_POS_X % plane;
    let warp = (UNIT_POS_X / plane) as usize;
    let row = CUBE_POS_X as usize;
    let base = row * d;

    // sum of squares in f32, one strided pass per thread
    let mut local = 0.0f32;
    let mut i = UNIT_POS_X;
    while i < d as u32 {
        let v = f32::cast_from(x_norm[base + i as usize]);
        local += v * v;
        i += CUBE_DIM_X;
    }
    let partial = plane_sum(local);
    // stage the per-warp partials, then combine. The trailing sync after the broadcast is the one that is
    // easy to leave out and impossible to notice, because it only races against the NEXT call that reuses the
    // slot (see csrc/kernels/gdn_layer.cu's block_sum).
    let mut staged = Shared::<[f32]>::new_slice(32usize);
    if lane == 0 {
        staged[warp] = partial;
    }
    sync_cube();
    let mut total = 0.0f32;
    if warp == 0 {
        let mut p = 0.0f32;
        if lane < warps as u32 {
            p = staged[lane as usize];
        }
        p = plane_sum(p);
        if lane == 0 {
            staged[0] = p;
        }
    }
    sync_cube();
    total = staged[0];
    sync_cube();

    let rstd = f32::inverse_sqrt(total / d as f32 + eps);
    let mut j = UNIT_POS_X;
    while j < d as u32 {
        let idx = base + (j as usize);
        let n = f32::cast_from(x_norm[idx]) * rstd;
        let w = 1.0f32 + f32::cast_from(weight[j as usize]);
        let g = f32::cast_from(gate[idx]);
        let silu = g / (1.0f32 + f32::exp(-g));
        let y = f32::cast_from(x_added[idx]) + n * w * silu;
        out[idx] = F::cast_from(y);
        j += CUBE_DIM_X;
    }
}

/// An empty kernel, to measure this backend's launch floor. The card's own floor measured from torch is
/// 1.13 us captured / 1.16 us eager, so anything above that is the language's marshalling.
#[cube(launch)]
pub fn nop(out: &mut [f32]) {
    if UNIT_POS_X == 0 && CUBE_POS_X == 0 {
        out[0] = 1.0;
    }
}

/// The same kernel with the SiLU removed, to price `f32::exp` on this backend -- it can lower to a slow
/// accurate path rather than the hardware unit.
#[cube(launch)]
pub fn rmsnorm_residual_nogate<F: Float, E: Float>(
    x_norm: &[F],
    x_added: &[F],
    weight: &[E],
    out: &mut [F],
    eps: f32,
    #[comptime] d: usize,
    #[comptime] warps: usize,
) {
    let plane = 32u32;
    let lane = UNIT_POS_X % plane;
    let warp = (UNIT_POS_X / plane) as usize;
    let row = CUBE_POS_X as usize;
    let base = row * d;
    let mut local = 0.0f32;
    let mut i = UNIT_POS_X;
    while i < d as u32 {
        let v = f32::cast_from(x_norm[base + i as usize]);
        local += v * v;
        i += CUBE_DIM_X;
    }
    let partial = plane_sum(local);
    let mut staged = Shared::<[f32]>::new_slice(32usize);
    if lane == 0 {
        staged[warp] = partial;
    }
    sync_cube();
    if warp == 0 {
        let mut p = 0.0f32;
        if lane < warps as u32 {
            p = staged[lane as usize];
        }
        p = plane_sum(p);
        if lane == 0 {
            staged[0] = p;
        }
    }
    sync_cube();
    let total = staged[0];
    sync_cube();
    let rstd = f32::inverse_sqrt(total / d as f32 + eps);
    let mut j = UNIT_POS_X;
    while j < d as u32 {
        let idx = base + j as usize;
        let n = f32::cast_from(x_norm[idx]) * rstd;
        let w = 1.0f32 + f32::cast_from(weight[j as usize]);
        out[idx] = F::cast_from(f32::cast_from(x_added[idx]) + n * w);
        j += CUBE_DIM_X;
    }
}

/// One buffer in, one out. If cubecl's captured launch cost scales with the number of buffer arguments, then a
/// fused forward should pack its activations into a few wide buffers rather than pass twenty narrow ones.
#[cube(launch)]
pub fn said1(a: &[f32], out: &mut [f32]) {
    let i = CUBE_POS_X as usize * CUBE_DIM_X as usize + UNIT_POS_X as usize;
    if i < out.len() {
        out[i] = a[i] * 2.0;
    }
}

#[cube(launch)]
pub fn said5(a: &[f32], b: &[f32], c: &[f32], d: &[f32], out: &mut [f32]) {
    let i = CUBE_POS_X as usize * CUBE_DIM_X as usize + UNIT_POS_X as usize;
    if i < out.len() {
        out[i] = a[i] + b[i] + c[i] + d[i];
    }
}

pub mod bridge;

/// Qwen3.5's `Qwen3_5RMSNorm` exactly, in one launch: `y = (x * rsqrt(mean(x^2) + eps)) * (1 + w)`.
///
/// This is the *real* call site shape. The residual add and the SiLU gate live outside the norm module in the
/// decoder layer (`x = x + input_layernorm(x)`), so a patch that replaces the norm module can only fuse the norm
/// itself -- 13 kernels down to 1, with the residual left as the torch op it already is.
///
/// Reference (`modeling_qwen3_5.py:835`): sum of squares and normalisation in f32, weight applied as `1.0 + w`,
/// ONE rounding to the model dtype at the end. `x` may be fp16 while the accumulation is f32, which is why the
/// two types are separate parameters.
#[cube(launch)]
pub fn rmsnorm_plain<F: Float, E: Float>(
    x: &[F],
    weight: &[E],
    out: &mut [F],
    eps: f32,
    #[comptime] d: usize,
    #[comptime] warps: usize,
) {
    let plane = 32u32;
    let lane = UNIT_POS_X % plane;
    let warp = (UNIT_POS_X / plane) as usize;
    let row = CUBE_POS_X as usize;
    let base = row * d;

    let mut local = 0.0f32;
    let mut i = UNIT_POS_X;
    while i < d as u32 {
        let v = f32::cast_from(x[base + i as usize]);
        local += v * v;
        i += CUBE_DIM_X;
    }
    let partial = plane_sum(local);
    let mut staged = Shared::<[f32]>::new_slice(32usize);
    if lane == 0 {
        staged[warp] = partial;
    }
    sync_cube();
    if warp == 0 {
        let mut p = 0.0f32;
        if lane < warps as u32 {
            p = staged[lane as usize];
        }
        p = plane_sum(p);
        if lane == 0 {
            staged[0] = p;
        }
    }
    sync_cube();
    // read into a register, THEN barrier: returning `staged[0]` directly races the next call's store (the same
    // defect gdn_layer.cu's block_sum had, memory 4248)
    let total = staged[0];
    sync_cube();

    let rstd = f32::inverse_sqrt(total / d as f32 + eps);
    let mut j = UNIT_POS_X;
    while j < d as u32 {
        let idx = base + j as usize;
        let n = f32::cast_from(x[idx]) * rstd;
        let w = 1.0f32 + f32::cast_from(weight[j as usize]);
        out[idx] = F::cast_from(n * w);
        j += CUBE_DIM_X;
    }
}
