"""AWS Trainium2 / Trainium3 through TorchNeuron (torch-neuronx), PyTorch's native Neuron backend.

TorchNeuron registers Trainium as the `neuron` device through PrivateUse1, so code written against
`torch.accelerator` runs on it unchanged. What differs from CUDA is declared below.

Status: the capabilities follow TorchNeuron's documented behaviour and are not yet validated on Trainium
hardware. Each value marked "unverified" defaults to the conservative choice, whose code path is the same
one the TPU platform already exercises.
"""

import torch
import torch_neuronx  # noqa: F401  registers the neuron device with torch

from miles.platforms import Platform


class NeuronPlatform(Platform):
    device_type = "neuron"
    # TorchNeuron's TorchDynamo backend (torch.compile(backend="neuron")), lowered by neuronx-cc
    compile_backend = "neuron"
    triton_kernels = False
    # TorchNeuron runs FlexAttention natively under torch.compile; the alternative path is the TPU splash kernel,
    # which needs jax and torch_tpu (unverified on hardware)
    flex_attention = True
    # unverified for the Neuron collective backend: the False paths fall back to all_reduce and plain SUM
    rooted_reduce = False
    premul_sum_reduce = False
    # a Neuron engine shares neither NCCL nor CUDA IPC with the trainer; disk-delta moves weights through host memory
    weight_transfer_modes = frozenset({"disk-delta"})
    # unverified whether the SGLang Neuron plugin reports per-prompt KV weight versions
    prefill_weight_versions = False
    # neuronx-cc compiles static graphs; the first TorchNeuron release does not support dynamic or symbolic shapes
    dynamic_shapes = False
    nvml_numa_affinity = False

    def bind_device_before_process_group(self, local_rank: int) -> None:
        # one NeuronCore per training rank, as torchrun lays TorchNeuron out (unverified on hardware)
        torch.accelerator.set_device_index(local_rank)

    def device_name(self) -> str | None:
        return None
