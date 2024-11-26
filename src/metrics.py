import torch
from utils import *
from fvcore.nn import FlopCountAnalysis


def measure_per_instance_inference_latency(model, device):
    """
    Measure the latency of the model per token.
    input:
        model: torch.nn.Module
        device: torch.device

    output:
        per_token_latency: float
    """

    avg_latency = 0

    # WRITE CODE HERE

    assert isinstance(avg_latency, float), f"avg_latency should be a float, but got {type(avg_latency)}"
    return avg_latency

def measure_per_batch_gpu_memory_consumption(model, batch_size, device):
    """
    Measure the GPU memory consumption of the model per batch.
    input:
        model: torch.nn.Module
        batch_size: int
        device: torch.device

    output:
        per_batch_gpu_memory: int
    """

    per_batch_gpu_memory = 0

    # WRITE CODE HERE

    assert isinstance(per_batch_gpu_memory, int), f"per_batch_gpu_memory should be an int, but got {type(per_batch_gpu_memory)}"
    return per_batch_gpu_memory