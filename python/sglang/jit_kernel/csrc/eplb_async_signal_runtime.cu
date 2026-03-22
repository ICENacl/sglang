#include "eplb_async_signal_runtime.h"

namespace sglang::eplb {

namespace {

__device__ __forceinline__ uint64_t load_signal(
    const EplbAsyncSignal* signal) {
  return *reinterpret_cast<const volatile uint64_t*>(&signal->step_and_owner);
}

__global__ void wait_signal_for_gpu_stage_kernel(
    EplbAsyncSignal* signal,
    int* enabled) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int enabled_value = 0;
  while (true) {
    const uint64_t signal_value = load_signal(signal);
    if (
        is_signal_disabled(signal_value) ||
        decode_signal_owner(signal_value) == kSignalOwnerGpu) {
      enabled_value = !is_signal_disabled(signal_value) &&
          !should_skip_signal_step(signal_value);
      break;
    }
  }
  *enabled = enabled_value;
}

__global__ void set_signal_for_cpu_stage_kernel(EplbAsyncSignal* signal) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  const uint64_t signal_value = load_signal(signal);
  if (is_signal_disabled(signal_value)) {
    return;
  }
  *reinterpret_cast<volatile uint64_t*>(&signal->step_and_owner) =
      signal_value | kSignalOwnerCpu;
  __threadfence_system();
}

}  // namespace

void wait_signal_for_gpu_stage_device(
    EplbAsyncSignal* signal,
    int* enabled,
    cudaStream_t stream) {
  wait_signal_for_gpu_stage_kernel<<<1, 1, 0, stream>>>(signal, enabled);
}

void set_signal_for_cpu_stage_device(
    EplbAsyncSignal* signal,
    cudaStream_t stream) {
  set_signal_for_cpu_stage_kernel<<<1, 1, 0, stream>>>(signal);
}

}  // namespace sglang::eplb
