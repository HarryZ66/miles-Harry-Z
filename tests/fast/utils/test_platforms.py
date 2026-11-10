import sys
import types

import pytest

from miles import platforms


def test_an_unknown_accelerator_is_refused(monkeypatch):
    """Any value but cuda, tpu or neuron would otherwise run the CUDA platform silently."""
    monkeypatch.setenv("MILES_ACCELERATOR", "TPU")
    with pytest.raises(ValueError, match="expected cuda, tpu or neuron"):
        platforms._select_platform()


def test_a_dynamic_helper_runs_eager_where_the_platform_compiles_static_shapes_only(monkeypatch):
    """The TPU compile backend rejects dynamic shapes, so the helper must stay the plain function there."""

    def helper(x):
        return x + 1

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", False)
    assert platforms.torch_compile(dynamic=True)(helper) is helper

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", True)
    assert platforms.torch_compile(dynamic=True)(helper) is not helper


@pytest.fixture
def neuron_platform(monkeypatch):
    """The Neuron platform, imported on a CPU host: a stub stands in for torch_neuronx, which only Trainium images ship."""
    monkeypatch.setitem(sys.modules, "torch_neuronx", types.ModuleType("torch_neuronx"))
    monkeypatch.delitem(sys.modules, "miles.platforms.neuron", raising=False)
    monkeypatch.setenv("MILES_ACCELERATOR", "neuron")
    return platforms._select_platform()


def test_neuron_selects_the_neuron_platform(neuron_platform):
    assert type(neuron_platform).__name__ == "NeuronPlatform"
    assert neuron_platform.device_type == "neuron"
    assert neuron_platform.compile_backend == "neuron"


def test_neuron_declares_static_shapes_and_no_cuda_only_features(neuron_platform):
    """neuronx-cc compiles static graphs, and Triton kernels and NVML exist only on CUDA."""
    assert neuron_platform.dynamic_shapes is False
    assert neuron_platform.triton_kernels is False
    assert neuron_platform.nvml_numa_affinity is False
    assert neuron_platform.device_name() is None


def test_neuron_keeps_flex_attention_so_torchtitan_does_not_load_the_tpu_kernel(neuron_platform):
    """Without FlexAttention, the torchtitan backend swaps in the TPU splash kernel, which needs jax and torch_tpu."""
    assert neuron_platform.flex_attention is True


def test_neuron_syncs_weights_only_through_disk_delta(neuron_platform):
    """A Neuron engine shares neither NCCL nor CUDA IPC with the trainer."""
    assert neuron_platform.weight_transfer_modes == frozenset({"disk-delta"})


def test_neuron_binds_each_rank_to_its_local_core_before_the_process_group(neuron_platform, monkeypatch):
    bound = []
    monkeypatch.setattr(platforms.torch.accelerator, "set_device_index", bound.append)
    neuron_platform.bind_device_before_process_group(3)
    neuron_platform.bind_device_after_process_group()
    assert bound == [3]
