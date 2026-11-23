//! NVPTX's packed int8 dot product.
//!
//! `dp4a` is one instruction on `sm_61` and later, and on a Pascal die it is the only fast arithmetic
//! there is: GP104 has no tensor cores and its fp16 is storage-only, so the alternatives are fp32 FMA
//! and this. Four signed byte products accumulate into one `i32` per instruction, per lane, per cycle.
//!
//! Until this lowering existed the LLVM backend fell back to the shared signed-byte polyfill -- four
//! multiplies and four adds -- which on a GTX 1070 runs at 11% of the instruction peak where the C++
//! backend reaches 103%.

use crate::prelude::*;
use cubecl_core::ir::dialect::math::Dp4aOp;

/// `dp4a.s32.s32` and `llvm.nvvm.idp4a.s.s` are the same instruction. cubecl's `dp4a` takes `i32` on
/// both sides, so only the signed/signed form is reachable from it; the other three exist in the
/// intrinsic for callers that carry `u8` halves.
const IDP4A: &str = "llvm.nvvm.idp4a.s.s";

/// The capability `dp4a` arrived with. Every later part keeps it.
pub(crate) const DP4A_ARCH: u32 = 61;

/// Lowers `dp4a` to the NVPTX intrinsic.
///
/// The intrinsic takes and returns signless `i32`, which is the type the frontend already gives the
/// operation, so nothing is converted on the way in or out.
pub(crate) fn dp4a(
    op: &Dp4aOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _info: &OperandsInfo,
) -> Result<()> {
    let a = op.a(ctx);
    let b = op.b(ctx);
    let c = op.c(ctx);
    let ty = a.get_type(ctx);
    let result = call_intrinsic(ctx, rw, IDP4A, ty, vec![a, b, c]);
    rw.replace_operation_with_values(ctx, op.get_operation(), vec![result]);
    Ok(())
}

/// Whether the architecture being compiled for has the instruction. An earlier die keeps the
/// polyfill -- which is correct everywhere, just slow.
pub(crate) fn supported(ctx: &Context) -> bool {
    ctx.sm_arch()
        .is_some_and(|arch| arch.version() >= DP4A_ARCH)
}
