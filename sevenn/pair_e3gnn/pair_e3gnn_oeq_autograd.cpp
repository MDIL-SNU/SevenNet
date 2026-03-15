/* -----------------------------------------------------------------------
   pair_e3gnn_oeq_autograd.cpp

   It registers the Autograd dispatch key for libtorch_tp_jit::jit_conv_forward
   so that pair_e3gnn can compute forces via autograd in pure C++ LibTorch.
   In Python, these registrations happen in:
       openequivariance._torch.TensorProductConv (jit_conv_forward)

   Contributing author: Jinmu You (SNU)
   ----------------------------------------------------------------------- */

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/core/DispatchKeySet.h>
#include <torch/library.h>
#include <torch/torch.h>

using namespace torch::autograd;

// ---------------------------------------------------------------------------
// Helper: call libtorch_tp_jit::jit_conv_backward through the dispatcher.
// Signature:
//   jit_conv_backward(Tensor json_bytes, int hash,
//                     Tensor L1_in, Tensor L2_in, Tensor W, Tensor L3_grad,
//                     Tensor rows, Tensor cols,
//                     Tensor workspace, Tensor transpose_perm)
//   -> (Tensor L1_grad, Tensor L2_grad, Tensor W_grad)
// ---------------------------------------------------------------------------
static std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_jit_conv_backward(
    const at::Tensor& json_bytes, int64_t hash,
    const at::Tensor& L1_in,     const at::Tensor& L2_in,
    const at::Tensor& W,         const at::Tensor& L3_grad,
    const at::Tensor& rows,      const at::Tensor& cols,
    const at::Tensor& workspace, const at::Tensor& transpose_perm)
{
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("libtorch_tp_jit::jit_conv_backward", "");
  c10::Stack stack;
  stack.reserve(10);
  stack.emplace_back(json_bytes);
  stack.emplace_back(hash);
  stack.emplace_back(L1_in);
  stack.emplace_back(L2_in);
  stack.emplace_back(W);
  stack.emplace_back(L3_grad);
  stack.emplace_back(rows);
  stack.emplace_back(cols);
  stack.emplace_back(workspace);
  stack.emplace_back(transpose_perm);

  // Exclude autograd keys so callBoxed goes straight to CUDA.
  c10::impl::ExcludeDispatchKeyGuard no_autograd(
      c10::autograd_dispatch_keyset);
  op.callBoxed(&stack);

  TORCH_INTERNAL_ASSERT(stack.size() == 3,
      "jit_conv_backward: expected 3 output tensors, got ", stack.size());
  return {stack[0].toTensor(), stack[1].toTensor(), stack[2].toTensor()};
}

// ---------------------------------------------------------------------------
// Custom autograd Function wrapping jit_conv_forward.
// ---------------------------------------------------------------------------
struct JitConvAutograd : public Function<JitConvAutograd> {
  static at::Tensor forward(
      AutogradContext* ctx,
      at::Tensor json_bytes, int64_t hash,
      at::Tensor L1_in,  at::Tensor L2_in,  at::Tensor W,
      int64_t L3_dim,
      at::Tensor rows,   at::Tensor cols,
      at::Tensor workspace, at::Tensor transpose_perm)
  {
    ctx->save_for_backward(
        {json_bytes, L1_in, L2_in, W, rows, cols, workspace, transpose_perm});
    ctx->saved_data["hash"]   = hash;
    ctx->saved_data["L3_dim"] = L3_dim;

    static auto fwd_op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("libtorch_tp_jit::jit_conv_forward", "");
    c10::Stack stack;
    stack.reserve(10);
    stack.emplace_back(json_bytes);
    stack.emplace_back(hash);
    stack.emplace_back(L1_in);
    stack.emplace_back(L2_in);
    stack.emplace_back(W);
    stack.emplace_back(L3_dim);
    stack.emplace_back(rows);
    stack.emplace_back(cols);
    stack.emplace_back(workspace);
    stack.emplace_back(transpose_perm);

    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
    fwd_op.callBoxed(&stack);

    TORCH_INTERNAL_ASSERT(stack.size() == 1,
        "jit_conv_forward: expected 1 output tensor, got ", stack.size());
    return stack[0].toTensor();
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
  {
    const auto& saved   = ctx->get_saved_variables();
    auto json_bytes     = saved[0];
    auto L1_in          = saved[1];
    auto L2_in          = saved[2];
    auto W              = saved[3];
    auto rows           = saved[4];
    auto cols           = saved[5];
    auto workspace      = saved[6];
    auto transpose_perm = saved[7];
    auto hash    = ctx->saved_data["hash"].toInt();
    auto L3_grad = grad_outputs[0];

    auto [L1_grad, L2_grad, W_grad] = dispatch_jit_conv_backward(
        json_bytes, hash, L1_in, L2_in, W, L3_grad,
        rows, cols, workspace, transpose_perm);

    return {
        at::Tensor(),  // json_bytes (no grad)
        at::Tensor(),  //  hash (no grad)
        L1_grad,       // L1_in
        L2_grad,       // L2_in
        W_grad,        // W
        at::Tensor(),  // L3_dim (no grad)
        at::Tensor(),  // rows (no grad)
        at::Tensor(),  // cols (no grad)
        at::Tensor(),  // workspace (no grad)
        at::Tensor(),  // transpose_perm (no grad)
    };
  }
};

// ---------------------------------------------------------------------------
// pair_e3gnn_oeq_register_autograd()
//
// Called from PairE3GNN::coeff() when "oeq: yes".
// Similar to Python's torch.library.register_autograd().
// ---------------------------------------------------------------------------
void pair_e3gnn_oeq_register_autograd()
{
  static torch::Library lib = [] {
    torch::Library l(
        torch::Library::IMPL,
        "libtorch_tp_jit",
        std::optional<c10::DispatchKey>(c10::DispatchKey::Autograd),
        __FILE__,
        __LINE__);
    l.impl(
        "jit_conv_forward",
        [](at::Tensor json_bytes, int64_t hash,
           at::Tensor L1_in, at::Tensor L2_in, at::Tensor W,
           int64_t L3_dim,
           at::Tensor rows,   at::Tensor cols,
           at::Tensor workspace, at::Tensor transpose_perm) -> at::Tensor {
          return JitConvAutograd::apply(
              json_bytes, hash, L1_in, L2_in, W, L3_dim,
              rows, cols, workspace, transpose_perm);
        });
    return std::move(l);
  }();
}
