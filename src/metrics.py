import torch
from utils import *
from fvcore.nn import FlopCountAnalysis


def measure_inference_latency_for_batch(model, batch_size, device):
    """
    Measure the latency of the model per batch.
    input:
        model: torch.nn.Module
        batch_size: int
        device: torch.device

    output:
        per_token_latency: float
    """
    dummy_input = torch.rand(batch_size, 3, 224, 224).to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        model(dummy_input)
        end_time.record()

        torch.cuda.synchronize()

        avg_latency = start_time.elapsed_time(end_time)

    avg_latency = avg_latency

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
    dummy_input = torch.rand(batch_size, 3, 224, 224).to(device)
    model = model.to(device)
    model.eval()

    # Reset GPU memory tracker
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        model(dummy_input)

    # Get the peak memory usage in bytes
    peak_memory = torch.cuda.max_memory_allocated(device)
    per_batch_gpu_memory = int(peak_memory / (1024 ** 2))  # Convert to MB

    assert isinstance(per_batch_gpu_memory, int), f"per_batch_gpu_memory should be an int, but got {type(per_batch_gpu_memory)}"
    return per_batch_gpu_memory

def FLOPs_per_instance(model, device):
    """
    Calculate the number of FLOPs for the model.
    input:
        model: torch.nn.Module
        device: torch.device

    output:
        flops: float
    """
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    model = model.to(device)
    model.eval()

    flops_analysis = FlopCountAnalysis(model, dummy_input)
    flops = flops_analysis.total()

    return flops