//! Inline PTX on NVPTX.
//!
//! `cube.asm` reaches the backends with its template already numbered: expansion has replaced each
//! `{}` with `$N`, outputs take `$0..` and inputs follow, which is the order the constraint list
//! states them in and so the order LLVM numbers the registers in.
//!
//! The C++ backend has always lowered this to `asm(...)` inside the generated CUDA C++. The LLVM
//! backend did not lower it at all, so a `gpu_asm!` kernel failed with "Operation cube.asm does not
//! have a conversion to LLVM instruction implemented" -- which meant a target could not reach an
//! instruction LLVM has no intrinsic for, on the one backend that is selected by default.

use crate::{
    nvptx::registers::{load_fragment, registers_result_ty, unpack_registers},
    prelude::*,
};
use cubecl_core::ir::dialect::{InlineAsmOp, MemoryClobbers};

/// Lowers `cube.asm` to an `llvm.inline_asm` call.
///
/// A `gpu_asm!` output is written once and read afterwards, so it takes an `=` constraint and is not
/// tied to an input: the template names it `$0` and the inputs `$1..`, which is the numbering
/// expansion produced. `InlineAsm`'s tied form is for a statement that rewrites a register in place
/// (`ldmatrix`), and using it here would shift every input by one and leave the template naming
/// registers that hold the wrong values.
pub(crate) fn inline_asm(
    op: &InlineAsmOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _info: &OperandsInfo,
) -> Result<()> {
    let template = op.asm(ctx).as_str().to_owned();
    let outputs = op.results(ctx);
    // A `mut` local reaches `gpu_asm!` as its slot, because a later statement may write through it:
    // the SSA value of `let mut d` is the address, not the number in it. The template wants the
    // number, so read it out. The result is stored back into the same slot by the statement that
    // wrote it, so nothing is written back here.
    let inputs = op
        .inputs(ctx)
        .into_iter()
        .map(|value| match pointee(ctx, value) {
            Some(elem) => load_fragment(ctx, rw, value, elem),
            None => value,
        })
        .collect::<Vec<_>>();

    let mut constraints: Vec<String> = outputs
        .iter()
        .map(|&value| format!("={}", register_class(ctx, value)))
        .collect();
    constraints.extend(
        inputs
            .iter()
            .map(|&value| register_class(ctx, value).to_string()),
    );
    if !matches!(op.memory_clobbers(ctx).clone(), MemoryClobbers::Nomem) {
        constraints.push("~{memory}".into());
    }

    // Arguments are the inputs only: LLVM's call already has one operand per output constraint
    // through `=`-prefixed entries, so the output value is the call's own result rather than an
    // argument. Passing it here makes the operation read what it defines.
    let args = inputs.clone();
    let result_ty = if outputs.is_empty() {
        VoidType::get(ctx).into()
    } else {
        registers_result_ty(
            ctx,
            outputs
                .iter()
                .map(|&it| cube_type_to_llvm(ctx, it.get_type(ctx)))
                .collect(),
        )
    };

    let asm = llvm::InlineAsmOp::new(
        ctx,
        result_ty,
        args,
        &template,
        &constraints.join(","),
        // A statement that claimed `pure` says what it does; one that did not may do anything.
        true,
    );
    if outputs.is_empty() {
        rw.insert_op(ctx, &asm);
        return Ok(());
    }
    rw.insert_op(ctx, &asm);
    let written = unpack_registers(ctx, rw, asm.get_result(ctx), outputs.len());
    rw.replace_operation_with_values(ctx, op.get_operation(), written);
    Ok(())
}

/// The PTX register class an operand is passed in, as the LLVM constraint letter.
fn register_class(ctx: &Context, value: Value) -> impl core::fmt::Display {
    let handle = value.get_type(ctx);
    let ty = handle.deref(ctx);
    // `cube.asm` reaches this backend after the type conversion, so a pointer is LLVM's opaque one.
    if ty.downcast_ref::<LlvmPointerType>().is_some()
        || ty.downcast_ref::<CubePointerType>().is_some()
    {
        return "l";
    }
    if ty.is::<FP32Type>() {
        return "f";
    }
    if ty.is::<FP64Type>() {
        return "d";
    }
    if ty.is::<FP16Type>() || ty.is::<BF16Type>() {
        return "h";
    }
    match ty.downcast_ref::<IntegerType>().map(|int| int.width()) {
        Some(16) => "h",
        Some(32) => "r",
        Some(64) => "l",
        _ => unreachable!("an inline PTX operand is a 16, 32 or 64-bit register"),
    }
}

/// The element type of a pointer operand, or `None` when the operand is already a value.
///
/// A `gpu_asm!` operand of a `cube` pointer arrives as LLVM's opaque pointer, which carries only an
/// address space, so the pointee comes from the cube type the load is standing in for. `i32` is what
/// every `gpu_asm!` in this tree writes its intermediate into; a wider one is reinterpreted by the
/// template rather than by the load.
fn pointee(ctx: &Context, value: Value) -> Option<TypeHandle> {
    let handle = value.get_type(ctx);
    handle.deref(ctx).downcast_ref::<LlvmPointerType>()?;
    Some(IntegerType::get(ctx, 32, Signedness::Signless).into())
}
