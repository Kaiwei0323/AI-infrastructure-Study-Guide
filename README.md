# AI-infrastructure-Study-Guide

Why we need GPU?
CPU vs GPU:
- CPU: Powerful generalist processor with a few big core, good at sequential tasks.
- GPU: Thousands of small core, good at parallel tasks.

CPU processing pipeline:
- CPU reads data from RAM -> CPU processes data -> CPU writes back to RAM

RAM: 
- Bandwidth: 64 GBps -> too slow for GPU throughput -> GPU keeps idle

Measure a GPU
- TFLOPS: Compute power -> Prefill: TTFT
- Bandwidth (TB / s): Load data / model speed -> Decode: TPOT
- Memory (VRAM): Store Model weights and KV cache

 EX:
 L40s
- TFLOPS (FP16): 362.05 tops
- Bandwidth: 864 GB / s
- Memory: 48 GB
 ADA6000
- TFLOPS (FP16): 728.5 tops
- Bandwidth: 960 GB / s
- Memory: 48 GB

Model Structure
- Safetensors: model weight
- Tokenizer: prompt -> token_ids
- 8B (model size): 16 GB (2x of parameter)

Model Metrics
- TTFT: Prefill
- TPOT / Token per second: Decode => Can be calculate with: 1 TB / s (VRAM Bandwidth) ÷ 16 GB (Model Size) = 62.5 token / s

