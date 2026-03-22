#include <sgl_kernel/utils.cuh>

#include <cuda_runtime_api.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int64_t kSignalOwnerGpu = 0;
constexpr int64_t kSignalOwnerCpu = 1;
constexpr int64_t kSignalDeviceMask = 0x1;

__global__ void eplb_wait_gpu_stage_kernel(int64_t* signal_step_and_owner) {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  const volatile int64_t* signal = reinterpret_cast<volatile int64_t*>(signal_step_and_owner);
  while ((*signal & kSignalDeviceMask) != kSignalOwnerGpu) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(100);
#endif
  }
  __threadfence_system();
}

__global__ void eplb_set_cpu_stage_kernel(int64_t* signal_step_and_owner) {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int64_t loaded = *signal_step_and_owner;
  *signal_step_and_owner = loaded | kSignalOwnerCpu;
  __threadfence_system();
}

auto get_signal_device_ptr(const tvm::ffi::TensorView signal_step_and_owner) -> int64_t* {
  void* device_ptr = nullptr;
  auto* host_ptr = static_cast<void*>(signal_step_and_owner.data_ptr());
  host::RuntimeCheck(host_ptr != nullptr, "signal_step_and_owner.data_ptr() must not be null.");
  host::RuntimeDeviceCheck(cudaHostGetDevicePointer(&device_ptr, host_ptr, 0));
  return static_cast<int64_t*>(device_ptr);
}

auto verify_signal_tensor_light(const tvm::ffi::TensorView signal_step_and_owner) -> void {
  using namespace host;
  const auto device = signal_step_and_owner.device();
  const auto dtype = signal_step_and_owner.dtype();
  RuntimeCheck(
      device.device_type == kDLCPU,
      "signal_step_and_owner must be a CPU tensor, but got device_type=",
      int(device.device_type));
  RuntimeCheck(
      dtype.code == kDLInt && dtype.bits == 64 && dtype.lanes == 1,
      "signal_step_and_owner must be int64, but got code=",
      int(dtype.code),
      " bits=",
      int(dtype.bits),
      " lanes=",
      int(dtype.lanes));
}

auto verify_stream_tensor_light(const tvm::ffi::TensorView stream_tensor) -> void {
  using namespace host;
  const auto device = stream_tensor.device();
  RuntimeCheck(
      device.device_type == kDLCUDA,
      "stream_tensor must be a CUDA tensor, but got device_type=",
      int(device.device_type));
  RuntimeCheck(stream_tensor.data_ptr() != nullptr, "stream_tensor.data_ptr() must not be null.");
}

auto eplb_wait_gpu_stage(
    const tvm::ffi::TensorView signal_step_and_owner,
    const tvm::ffi::TensorView stream_tensor) -> void {
  verify_signal_tensor_light(signal_step_and_owner);
  verify_stream_tensor_light(stream_tensor);

  auto* signal_ptr = get_signal_device_ptr(signal_step_and_owner);
  const auto stream = host::LaunchKernel::resolve_device(stream_tensor.device());
  eplb_wait_gpu_stage_kernel<<<1, 1, 0, stream>>>(signal_ptr);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

auto eplb_set_cpu_stage(
    const tvm::ffi::TensorView signal_step_and_owner,
    const tvm::ffi::TensorView stream_tensor) -> void {
  verify_signal_tensor_light(signal_step_and_owner);
  verify_stream_tensor_light(stream_tensor);

  auto* signal_ptr = get_signal_device_ptr(signal_step_and_owner);
  const auto stream = host::LaunchKernel::resolve_device(stream_tensor.device());
  eplb_set_cpu_stage_kernel<<<1, 1, 0, stream>>>(signal_ptr);
  host::RuntimeDeviceCheck(cudaGetLastError());
}

}  // namespace
