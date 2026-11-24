#ifndef SEVENN_E3GNN_GRAPH_STAGE_H
#define SEVENN_E3GNN_GRAPH_STAGE_H

#include <torch/script.h>
#include <torch/torch.h>
#include <array>
#include <memory>

#ifdef SEVENN_E3GNN_CUDA_GRAPH
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace LAMMPS_NS {

class E3GNNGraphStage {
  using Dict = c10::impl::GenericDict;
  using Tensors = std::vector<torch::Tensor>;
  using Shape = std::array<int64_t, 3>;

  struct Pass {
    Dict output;
    Tensors wrt;
    Tensors targets;
  };

  static Pass run(torch::jit::Module &module, Dict inputs, bool first, bool last) {
    Tensors wrt{inputs.at("edge_vec").toTensor()};
    if (!first) {
      wrt.push_back(inputs.at("x").toTensor());
      wrt.push_back(inputs.at("self_cont_tmp").toTensor());
      wrt.push_back(inputs.at("x_ghost").toTensor());
    }
    auto output = module.forward({inputs.copy()}).toGenericDict();
    Tensors targets;
    if (last) {
      targets.push_back(output.at("inferred_total_energy").toTensor().squeeze());
    } else {
      targets.push_back(output.at("x").toTensor());
      targets.push_back(output.at("self_cont_tmp").toTensor());
    }
    return {output, wrt, targets};
  }

  static Dict prepare(const Dict &input, bool first, bool own_storage) {
    Dict tensors(c10::StringType::get(), c10::TensorType::get());
    for (const auto &item : input) {
      auto name = item.key().toStringRef();
      auto value = item.value().toTensor().detach();
      if (own_storage) value = value.clone();
      if (name == "edge_vec" || (!first && (name == "x" ||
          name == "self_cont_tmp" || name == "x_ghost"))) {
        value.set_requires_grad(true);
      }
      tensors.insert(item.key(), value);
    }
    return tensors;
  }

#ifdef SEVENN_E3GNN_CUDA_GRAPH
  struct Capture {
    explicit Capture(Dict inputs) : inputs(std::move(inputs)) {}

    Dict inputs;
    at::cuda::CUDAGraph forward;
    at::cuda::CUDAGraph backward;
    std::unique_ptr<Pass> pass;
    Tensors seeds;
    Tensors gradients;
  };

  template <typename Call>
  static void capture(at::cuda::CUDAGraph &graph, Call call,
                      at::cuda::MempoolId_t pool = {0, 0}) {
    graph.capture_begin(pool);
    try {
      call();
    } catch (...) {
      try { graph.capture_end(); } catch (...) {}
      throw;
    }
    graph.capture_end();
  }

  void record(torch::jit::Module &module, const Dict &inputs, bool first, bool last) {
    auto current = c10::cuda::getCurrentCUDAStream(
        inputs.at("edge_vec").toTensor().get_device());
    auto side = c10::cuda::getStreamFromPool(false, current.device_index());
    at::cuda::CUDAEvent ready;
    ready.record(current);
    ready.block(side);
    c10::cuda::CUDAStreamGuard guard(side);

    auto candidate = std::make_unique<Capture>(prepare(inputs, first, true));
    auto &entry = *candidate;
    {
      auto warmup = run(module, entry.inputs, first, last);
      if (!last) {
        for (const auto &target : warmup.targets)
          entry.seeds.push_back(torch::zeros_like(target));
      }
      torch::autograd::grad(warmup.targets, warmup.wrt, entry.seeds);
    }
    try {
      capture(entry.forward, [&] {
        entry.pass = std::make_unique<Pass>(run(module, entry.inputs, first, last));
      });
      // The forward replay refreshes the activations read by the local VJP.
      capture(entry.backward, [&] {
        entry.gradients = torch::autograd::grad(
            entry.pass->targets, entry.pass->wrt, entry.seeds);
      }, entry.forward.pool());
    } catch (...) {
      ready.record(side);
      ready.block(current);
      throw;
    }
    ready.record(side);
    ready.block(current);
    captured_ = std::move(candidate);
    ++captures_;
  }

  bool compatible(const Dict &input) const {
    const auto &stored = captured_->inputs;
    if (input.size() != stored.size()) return false;
    for (const auto &item : input) {
      if (!stored.contains(item.key())) return false;
      const auto a = stored.at(item.key()).toTensor();
      const auto b = item.value().toTensor();
      if (a.sizes() != b.sizes() || a.options().dtype() != b.options().dtype() ||
          a.device() != b.device()) return false;
    }
    return true;
  }

  std::unique_ptr<Capture> captured_;
  Shape last_shape_{-1, -1, -1};
#endif

  std::unique_ptr<Pass> eager_;
  bool use_capture_ = false;
  int64_t captures_ = 0;
  int64_t replays_ = 0;
  std::string failure_;

public:
  static constexpr bool available() {
#ifdef SEVENN_E3GNN_CUDA_GRAPH
    return true;
#else
    return false;
#endif
  }

  Dict forward(torch::jit::Module &module, const Dict &inputs, bool first, bool last) {
    use_capture_ = false;
#ifdef SEVENN_E3GNN_CUDA_GRAPH
    const auto geometry = inputs.at("edge_vec").toTensor();
    if (geometry.is_cuda() && geometry.size(0) != 0 && failure_.empty()) {
      const Shape shape{
          inputs.at("x").toTensor().size(0),
          inputs.contains("x_ghost") ? inputs.at("x_ghost").toTensor().size(0) : 0,
          geometry.size(0)};
      if (captured_ && (shape != last_shape_ || !compatible(inputs)))
        captured_.reset();
      if (!captured_ && shape == last_shape_) {
        try {
          record(module, inputs, first, last);
        } catch (const std::exception &e) {
          failure_ = e.what();
        }
      }
      last_shape_ = shape;
      if (captured_) {
        torch::NoGradGuard no_grad;
        for (const auto &item : inputs)
          captured_->inputs.at(item.key()).toTensor().copy_(item.value().toTensor());
        captured_->forward.replay();
        use_capture_ = true;
        return captured_->pass->output.copy();
      }
    }
#endif
    eager_ = std::make_unique<Pass>(run(module, prepare(inputs, first, false), first, last));
    return eager_->output.copy();
  }

  Tensors backward(const Tensors &seeds) {
#ifdef SEVENN_E3GNN_CUDA_GRAPH
    if (use_capture_) {
      torch::NoGradGuard no_grad;
      TORCH_CHECK(seeds.size() == captured_->seeds.size(),
                  "e3gnn CUDA Graph received the wrong number of gradient seeds");
      for (size_t i = 0; i < seeds.size(); ++i)
        captured_->seeds[i].copy_(seeds[i]);
      captured_->backward.replay();
      ++replays_;
      return captured_->gradients;
    }
#endif
    auto grads = torch::autograd::grad(eager_->targets, eager_->wrt, seeds);
    eager_.reset();
    return grads;
  }

  int64_t captures() const { return captures_; }
  int64_t replays() const { return replays_; }
  const std::string &failure() const { return failure_; }
};
} // namespace LAMMPS_NS

#endif
