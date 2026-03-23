import atexit
import ctypes
import contextlib
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_world_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

logger = logging.getLogger(__name__)


def _sanitize_name(raw: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", raw)


def _get_host_mirror_model_name(model_config) -> str:
    architectures = getattr(model_config.hf_config, "architectures", None)
    if architectures and len(architectures) > 0:
        return architectures[0]
    return os.path.basename(model_config.model_path.rstrip("/")) or model_config.model_path


def _element_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _compute_node_owner_physical_ids(
    physical_to_logical_map: torch.Tensor,
    *,
    node_physical_start: int,
    node_physical_end: int,
    num_logical_experts: int,
) -> list[int]:
    owners = [-1] * num_logical_experts
    for global_physical_expert_id in range(node_physical_start, node_physical_end):
        logical_expert_id = int(physical_to_logical_map[global_physical_expert_id].item())
        if owners[logical_expert_id] == -1:
            owners[logical_expert_id] = global_physical_expert_id
    return owners


def _build_cross_node_transfer_plan(
    availability: torch.Tensor,
) -> list[tuple[int, int, int]]:
    plan = []
    num_nodes, num_logical_experts = availability.shape
    for logical_expert_id in range(num_logical_experts):
        src_nodes = torch.nonzero(availability[:, logical_expert_id], as_tuple=False).flatten()
        if src_nodes.numel() == 0:
            continue
        src_node_rank = int(src_nodes[0].item())
        for dst_node_rank in range(num_nodes):
            if int(availability[dst_node_rank, logical_expert_id].item()) == 0:
                plan.append((logical_expert_id, src_node_rank, dst_node_rank))
    return plan


@dataclass
class _ShmRecord:
    shm: shared_memory.SharedMemory
    tensor: torch.Tensor
    name: str
    is_owner: bool
    unlink_on_close: bool


class EPLBAsyncHostMirrorManager:
    def __init__(self, server_args, model_config):
        config = ModelConfigForExpertLocation.from_model_config(model_config)
        assert config is not None, "EPLB async requires a MoE model."

        self._server_args = server_args
        self._world_group = get_world_group()
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._local_rank = self._world_group.local_rank
        self._local_world_size = self._world_group.local_size or max(
            1, self._world_size // max(1, server_args.nnodes)
        )
        self._node_rank = server_args.node_rank
        self._num_nodes = server_args.nnodes
        self._is_owner = self._local_rank == 0
        self._num_logical_experts = config.num_logical_experts
        self._records: dict[tuple[int, int], _ShmRecord] = {}
        self._valid_records: dict[int, _ShmRecord] = {}
        self._layer_tensors: dict[int, list[torch.Tensor]] = {}
        self._layer_num_tensors: dict[int, int] = {}
        self._staging_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._leader_device_group = None
        self._leader_cpu_group = None
        self._leader_global_ranks: list[int] = []
        self._closed = False
        self._registered_atexit = False
        self._lock = threading.Lock()
        self._reuse_existing_shm = envs.SGLANG_EPLB_ASYNC_HOST_MIRROR_REUSE_SHM.get()

        model_name = _sanitize_name(_get_host_mirror_model_name(model_config))
        master_port = _sanitize_name(os.environ.get("MASTER_PORT", "0"))
        self._base_name = (
            f"sglang_eplb_async_{model_name}_p{master_port}_n{server_args.node_rank}"
        )
        self._maybe_create_leader_groups()

    def build_from_loaded_model(self, routed_experts_weights_of_layer) -> None:
        metadata = get_global_expert_location_metadata()
        assert metadata is not None, "EPLB async host mirror requires expert metadata."

        self._create_records(routed_experts_weights_of_layer)
        if self._can_reuse_existing_data():
            logger.info("Reusing existing EPLB async host mirror shared memory data.")
        else:
            self._populate_local_node_shards(
                routed_experts_weights_of_layer=routed_experts_weights_of_layer,
                metadata=metadata,
            )
            self._barrier_all_ranks()
            self._fill_missing_from_remote_nodes()
            self._barrier_all_ranks()
        self._validate_completeness()

        for layer_id, tensors in routed_experts_weights_of_layer.items():
            attached = []
            for tensor_index, _ in enumerate(tensors):
                record = self._records.get((layer_id, tensor_index))
                if record is None:
                    raise RuntimeError(
                        f"EPLB async host mirror is incomplete for layer={layer_id} tensor_index={tensor_index}."
                    )
                attached.append(record.tensor)
            self._layer_tensors[layer_id] = attached

        self._maybe_register_atexit()

    def get_expert_tensors(self, layer_id: int, logical_expert_id: int):
        if layer_id not in self._layer_tensors:
            raise KeyError(f"Layer {layer_id} does not exist in async host mirror.")
        return [tensor[logical_expert_id] for tensor in self._layer_tensors[layer_id]]

    def close(self):
        if self._closed:
            return
        self._closed = True

        for record in self._records.values():
            with contextlib.suppress(Exception):
                record.shm.close()
            if record.unlink_on_close:
                with contextlib.suppress(FileNotFoundError):
                    record.shm.unlink()
        for record in self._valid_records.values():
            with contextlib.suppress(Exception):
                record.shm.close()
            if record.unlink_on_close:
                with contextlib.suppress(FileNotFoundError):
                    record.shm.unlink()
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                if self._leader_device_group is not None:
                    dist.destroy_process_group(self._leader_device_group)
            with contextlib.suppress(Exception):
                if self._leader_cpu_group is not None:
                    dist.destroy_process_group(self._leader_cpu_group)

        self._records.clear()
        self._valid_records.clear()
        self._layer_tensors.clear()
        self._layer_num_tensors.clear()
        self._staging_buffers.clear()
        self._leader_device_group = None
        self._leader_cpu_group = None

    def _maybe_register_atexit(self):
        if self._registered_atexit:
            return
        self._registered_atexit = True
        atexit.register(self.close)

    def _create_records(self, routed_experts_weights_of_layer) -> None:
        for layer_id, tensors in routed_experts_weights_of_layer.items():
            self._layer_num_tensors[layer_id] = len(tensors)
        if self._is_owner:
            for layer_id, tensors in routed_experts_weights_of_layer.items():
                self._get_or_create_valid_tensor(layer_id)
                for tensor_index, tensor in enumerate(tensors):
                    self._get_or_create_tensor(
                        layer_id=layer_id,
                        tensor_index=tensor_index,
                        sample_tensor=tensor,
                    )
        self._barrier_all_ranks()
        if not self._is_owner:
            for layer_id, tensors in routed_experts_weights_of_layer.items():
                self._get_or_create_valid_tensor(layer_id)
                for tensor_index, tensor in enumerate(tensors):
                    self._get_or_create_tensor(
                        layer_id=layer_id,
                        tensor_index=tensor_index,
                        sample_tensor=tensor,
                    )
        self._barrier_all_ranks()

    def _populate_local_node_shards(self, *, routed_experts_weights_of_layer, metadata) -> None:
        num_local_physical_experts = metadata.num_local_physical_experts
        local_physical_start = self._rank * num_local_physical_experts
        node_physical_start = self._node_rank * self._local_world_size * num_local_physical_experts
        node_physical_end = node_physical_start + self._local_world_size * num_local_physical_experts

        for layer_id, tensors in routed_experts_weights_of_layer.items():
            node_owner_physical_ids = _compute_node_owner_physical_ids(
                metadata.physical_to_logical_map_cpu[layer_id],
                node_physical_start=node_physical_start,
                node_physical_end=node_physical_end,
                num_logical_experts=self._num_logical_experts,
            )
            valid_tensor = self._valid_records[layer_id].tensor

            for local_expert_id in range(num_local_physical_experts):
                global_physical_expert_id = local_physical_start + local_expert_id
                logical_expert_id = int(
                    metadata.physical_to_logical_map_cpu[
                        layer_id, global_physical_expert_id
                    ].item()
                )
                if node_owner_physical_ids[logical_expert_id] != global_physical_expert_id:
                    continue

                for tensor_index, tensor in enumerate(tensors):
                    record = self._records[(layer_id, tensor_index)]
                    record.tensor[logical_expert_id].copy_(tensor[local_expert_id].cpu())
                valid_tensor[logical_expert_id] = 1

    def _fill_missing_from_remote_nodes(self) -> None:
        if self._leader_cpu_group is None:
            return
        if not self._is_owner:
            return

        for layer_id in sorted(self._valid_records):
            local_bitmap = self._valid_records[layer_id].tensor.clone()
            gathered = [torch.empty_like(local_bitmap) for _ in range(self._num_nodes)]
            dist.all_gather(gathered, local_bitmap, group=self._leader_cpu_group)
            availability = torch.stack(gathered, dim=0)
            transfer_plan = _build_cross_node_transfer_plan(availability)
            if len(transfer_plan) == 0:
                continue

            for logical_expert_id, src_node_rank, dst_node_rank in transfer_plan:
                if self._node_rank not in (src_node_rank, dst_node_rank):
                    continue
                self._transfer_logical_expert(
                    layer_id=layer_id,
                    logical_expert_id=logical_expert_id,
                    src_node_rank=src_node_rank,
                    dst_node_rank=dst_node_rank,
                )

            dist.barrier(group=self._leader_cpu_group)

    def _transfer_logical_expert(
        self,
        *,
        layer_id: int,
        logical_expert_id: int,
        src_node_rank: int,
        dst_node_rank: int,
    ) -> None:
        assert self._leader_global_ranks
        src_global_rank = self._leader_global_ranks[src_node_rank]
        dst_global_rank = self._leader_global_ranks[dst_node_rank]

        if self._node_rank == dst_node_rank:
            valid_tensor = self._valid_records[layer_id].tensor
        else:
            valid_tensor = None

        for tensor_index in range(self._layer_num_tensors[layer_id]):
            record = self._records[(layer_id, tensor_index)]
            staging = self._staging_buffers[(layer_id, tensor_index)]
            if self._node_rank == src_node_rank:
                staging.copy_(record.tensor[logical_expert_id], non_blocking=False)
                self._send_tensor(staging, dst_global_rank)
            elif self._node_rank == dst_node_rank:
                self._recv_tensor(staging, src_global_rank)
                record.tensor[logical_expert_id].copy_(staging, non_blocking=False)

        if valid_tensor is not None:
            valid_tensor[logical_expert_id] = 1

    def _send_tensor(self, tensor: torch.Tensor, dst_global_rank: int) -> None:
        if tensor.is_cuda:
            dist.send(tensor, dst=dst_global_rank, group=self._leader_device_group)
        else:
            dist.send(tensor, dst=dst_global_rank, group=self._leader_cpu_group)

    def _recv_tensor(self, tensor: torch.Tensor, src_global_rank: int) -> None:
        if tensor.is_cuda:
            dist.recv(tensor, src=src_global_rank, group=self._leader_device_group)
        else:
            dist.recv(tensor, src=src_global_rank, group=self._leader_cpu_group)

    def _validate_completeness(self) -> None:
        for layer_id, valid_record in self._valid_records.items():
            missing = torch.nonzero(valid_record.tensor == 0, as_tuple=False).flatten().tolist()
            if missing:
                raise RuntimeError(
                    "EPLB async host mirror is incomplete after initialization: "
                    f"node_rank={self._node_rank} layer_id={layer_id} missing_logical_expert_ids={missing}"
                )

    def _barrier_all_ranks(self) -> None:
        if dist.is_initialized():
            self._world_group.barrier()

    def _maybe_create_leader_groups(self) -> None:
        if (not dist.is_initialized()) or self._num_nodes <= 1:
            return

        self._leader_global_ranks = [
            node_rank * self._local_world_size for node_rank in range(self._num_nodes)
        ]
        backend = dist.get_backend(self._world_group.device_group)
        self._leader_device_group = dist.new_group(
            self._leader_global_ranks,
            backend=backend,
        )
        self._leader_cpu_group = dist.new_group(
            self._leader_global_ranks,
            backend="gloo",
        )

    def _get_or_create_tensor(
        self,
        *,
        layer_id: int,
        tensor_index: int,
        sample_tensor: torch.Tensor,
    ) -> torch.Tensor:
        key = (layer_id, tensor_index)
        if key in self._records:
            return self._records[key].tensor

        shm_name = self._tensor_shm_name(layer_id, tensor_index)
        num_bytes = self._num_bytes_for_tensor(sample_tensor)
        shm, is_owner, unlink_on_close, created_new = self._open_or_create_shm(
            shm_name, num_bytes
        )
        tensor = _create_buffer_tensor(
            shm=shm,
            shape=self._mirror_shape(sample_tensor),
            dtype=sample_tensor.dtype,
        )
        if created_new:
            tensor.zero_()
        _cuda_host_register_tensor(tensor)
        self._records[key] = _ShmRecord(
            shm=shm,
            tensor=tensor,
            name=shm_name,
            is_owner=is_owner,
            unlink_on_close=unlink_on_close,
        )
        self._staging_buffers[key] = torch.empty(
            tensor.shape[1:],
            dtype=sample_tensor.dtype,
            device=sample_tensor.device,
        )
        return tensor

    def _get_or_create_valid_tensor(self, layer_id: int) -> torch.Tensor:
        if layer_id in self._valid_records:
            return self._valid_records[layer_id].tensor

        shm_name = self._valid_shm_name(layer_id)
        num_bytes = self._num_logical_experts * _element_size(torch.uint8)
        shm, is_owner, unlink_on_close, created_new = self._open_or_create_shm(
            shm_name, num_bytes
        )
        tensor = _create_buffer_tensor(
            shm=shm,
            shape=(self._num_logical_experts,),
            dtype=torch.uint8,
        )
        if created_new:
            tensor.zero_()
        self._valid_records[layer_id] = _ShmRecord(
            shm=shm,
            tensor=tensor,
            name=shm_name,
            is_owner=is_owner,
            unlink_on_close=unlink_on_close,
        )
        return tensor

    def _mirror_shape(self, tensor: torch.Tensor):
        return (self._num_logical_experts, *tensor.shape[1:])

    def _num_bytes_for_tensor(self, tensor: torch.Tensor) -> int:
        return math.prod(self._mirror_shape(tensor)) * _element_size(tensor.dtype)

    def _tensor_shm_name(self, layer_id: int, tensor_index: int) -> str:
        return f"{self._base_name}_l{layer_id}_t{tensor_index}"

    def _valid_shm_name(self, layer_id: int) -> str:
        return f"{self._base_name}_l{layer_id}_valid"

    def _open_or_create_shm(self, shm_name: str, num_bytes: int):
        if self._is_owner:
            return self._open_or_create_owner_shm(shm_name, num_bytes)

        for _ in range(200):
            try:
                shm = _open_shared_memory(name=shm_name, track=False)
                self._validate_shm_size(shm, shm_name, num_bytes)
                return shm, False, False, False
            except FileNotFoundError:
                time.sleep(0.05)

        raise RuntimeError(f"Timed out waiting for host mirror shared memory {shm_name}.")

    def _open_or_create_owner_shm(self, shm_name: str, num_bytes: int):
        try:
            shm = _open_shared_memory(
                name=shm_name,
                create=True,
                size=num_bytes,
                track=not self._reuse_existing_shm,
            )
            return shm, True, not self._reuse_existing_shm, True
        except FileExistsError:
            if self._reuse_existing_shm:
                logger.info(
                    "Reusing existing EPLB async shared memory name: %s",
                    shm_name,
                )
                shm = _open_shared_memory(name=shm_name, track=False)
                self._validate_shm_size(shm, shm_name, num_bytes)
                return shm, True, False, False
            logger.warning(
                "Found existing EPLB async shared memory name: %s, unlinking...",
                shm_name,
            )
            stale = shared_memory.SharedMemory(name=shm_name)
            stale.close()
            stale.unlink()
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_bytes)
            return shm, True, True, True

    def _validate_shm_size(self, shm, shm_name: str, num_bytes: int):
        actual_size = getattr(shm, "size", None)
        if actual_size is not None and actual_size != num_bytes:
            shm.close()
            raise RuntimeError(
                f"EPLB async shared memory size mismatch for {shm_name}: "
                f"expected={num_bytes} actual={actual_size}"
            )

    def _can_reuse_existing_data(self) -> bool:
        if not self._reuse_existing_shm:
            return False
        if len(self._valid_records) == 0:
            return False
        for valid_record in self._valid_records.values():
            if not torch.all(valid_record.tensor == 1):
                return False
        return True


def _create_buffer_tensor(
    *,
    shm: shared_memory.SharedMemory,
    shape,
    dtype: torch.dtype,
) -> torch.Tensor:
    numel = math.prod(shape)
    raw = torch.frombuffer(shm.buf, dtype=torch.uint8)
    return raw.view(dtype)[:numel].view(*shape)


def _open_shared_memory(*, name: str, create: bool = False, size: int = 0, track: bool = True):
    if track:
        return shared_memory.SharedMemory(name=name, create=create, size=size)
    with patch.object(resource_tracker, "register", lambda *args, **kwargs: None):
        return shared_memory.SharedMemory(name=name, create=create, size=size)


def _cuda_host_register_tensor(tensor: torch.Tensor):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "EPLB async requires CUDA, but torch.cuda.is_available() is False."
        )

    libcudart = ctypes.CDLL("libcudart.so")
    cuda_host_register_portable = 0x01
    num_bytes = tensor.numel() * tensor.element_size()
    result = libcudart.cudaHostRegister(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_size_t(num_bytes),
        ctypes.c_uint(cuda_host_register_portable),
    )
    if result != 0:
        raise RuntimeError(
            f"cudaHostRegister failed with CUDA error code {result} "
            f"for tensor shape={tuple(tensor.shape)} dtype={tensor.dtype}"
        )


_INSTANCE = None


def get_global_eplb_async_host_mirror_manager():
    return _INSTANCE


def set_global_eplb_async_host_mirror_manager(value):
    global _INSTANCE
    _INSTANCE = value
