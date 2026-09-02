# AI-infrastructure-Study-Guide

## Why we need GPU?

### CPU vs GPU

- CPU: Powerful generalist processor with a few big cores, good at sequential tasks.
- GPU: Thousands of small cores, good at parallel tasks.

### CPU processing pipeline

- CPU reads data from RAM -> CPU processes data -> CPU writes back to RAM

### RAM

- Bandwidth: 64 GBps -> too slow for GPU throughput -> GPU stays idle

### GPU Memory Hierarchy

| Memory type | CUDA syntax | Location | Scope | Speed | Size |
| --- | --- | --- | --- | --- | --- |
| Registers | plain local var, e.g. `float x;` | On-chip (inside SM) | 1 thread only | Fastest | ~255 regs/thread |
| Local memory | large/indexed local arrays (compiler spills automatically) | Off-chip (physically same as global) | 1 thread only | Slow | - |
| Shared memory (SRAM) | `__shared__ float buf[N];` | On-chip (inside SM) | All threads in 1 block | Very fast | ~48–228 KB / SM |
| Global memory (VRAM / HBM / GDDR) | `__device__ float x;` or `cudaMalloc()` pointers | Off-chip | All threads, all blocks, host | Slow (relatively) | GBs |
| Constant memory | `__constant__ float x;` | Off-chip, cached on-chip | All threads, read-only | Fast if cached | 64 KB |

## Measure a GPU

- TFLOPS: Compute power -> Prefill: TTFT
- Bandwidth (TB / s): Load data / model speed -> Decode: TPOT
- Memory (VRAM): Store model weights and KV cache

### EX: L40s vs ADA6000

| | L40s | ADA6000 |
| --- | --- | --- |
| TFLOPS (FP16) | 362.05 tops | 728.5 tops |
| Bandwidth | 864 GB / s | 960 GB / s |
| Memory | 48 GB | 48 GB |

## Self Attention

### Why we need self-attention?

Old model (RNN + Encoder / Decoder) drawback: Process tokens serially, and the old context will be forgotten and the new context will have more weight.

- Training: output can be processed in parallel
- Inference: output one at a time

### Encoding

`Input -> Embedding -> Multi-head attention`

- Input_embedding = text embedding + text position embedding (`n x 512`)
- Multi-head attention: `wQ + wK + wV` (`512 x 512`)

### Q, K, V

- Q: Ask a question?
- K: Yes or No
- V: Actual meaning

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

- Single head = Input_embedding (`n x 512`) * (`wQ` (`512 x 512`) + `wK` (`512 x 512`) + `wV` (`512 x 512`)) = `n x 512`, Attention(Q, K, V) = output
- Multi-head = Input_embedding (`n x 512`) * 8 * (`wQ` (`512 x 64`) + `wK` (`512 x 64`) + `wV` (`512 x 64`)) = `8 x (n x 64)` -> linear layer -> `n x 512`, Attention(Q, K, V) = output
    - Making `8 x (QKV)` is like feature extraction to make it more accurate

### MHA vs GQA vs MQA

| | Q heads | KV heads |
| --- | --- | --- |
| MHA | 8 | 8 |
| GQA | 32 | 8 (Llama 3 8B) |
| MQA | many | 1 |

- Multiple Q heads can share one KV head: Q is "how I look up", K/V is the cached content. KV cache size uses `num_kv_heads`, not Q heads.

### Decoder

`output -> embedding -> Masked multi-head attention -> multi-head attention -> softmax -> output`


## Model Structure

- Safetensors: model weights
- Tokenizer: prompt -> token_ids
- 8B (model size): 16 GB (2x the parameters)

## Model Metrics

- TTFT: Prefill
- TPOT / Token per second: Decode => Can be calculated with: `1 TB / s (VRAM Bandwidth) ÷ 16 GB (Model Size) = 62.5 token / s`

## VLLM backend (Single node)

### PageAttention

- a logical block table maps a logical block to a physical block

#### Block Calculation

- Each token: `2 (K & V) x num_layers (32) x num_kv_heads (8) x head_dim (128) x dtype_size (2) = 128 KB / token`
- Block Size: `128 x 16` (16 tokens per block)
- Available KV cache memory = `90% x total memory - model_weights - activation memory`
- Num of blocks = `Available KV cache memory ÷ Block Size`

#### CUDA Kernel Optimization

1. logical block maps to a physical block, calculate KV cache and write into physical memory
2. GQA (Grouped Query Attention) 8 Q heads grouped to 2 KV heads
3. Softmax: Track the max value, scale the rest
    - naive: save all values then softmax, space complexity O(n)
    - Optimized: only save the max value, space complexity O(1)
4. Partition long context and hand it to multiple thread blocks to process in parallel and then aggregate to softmax
5. Prefill leverages FlashAttention kernel
    - FlashAttention:
        - Tiling: split QKV into small pieces and load to SRAM to compute, Q (VRAM -> Register), KV (VRAM -> SRAM)
        1. `S = Q Kᵀ`
        2. `P = softmax(S)`
        3. `O = P V`
```cuda-cpp
__shared__ float Ks[BC][D_MAX];   // SRAM tile for K
__shared__ float Vs[BC][D_MAX];   // SRAM tile for V

// ---- Load K AND V tiles: VRAM -> SRAM ----
for (int r = tid; r < kv_rows_this_tile; r += BR) {
    for (int d = 0; d < head_dim; d++) {
        Ks[r][d] = K[(kv_tile_start + r) * head_dim + d];  // VRAM -> SRAM
        Vs[r][d] = V[(kv_tile_start + r) * head_dim + d];  // VRAM -> SRAM
    }
}
__syncthreads();  // wait until the whole tile (K and V) is loaded before anyone reads it

int row = q_tile_start + tid;   // each thread owns exactly one Q row for the whole kernel

float q_row[D_MAX];             // plain array -> lives in registers (per-thread)
for (int d = 0; d < head_dim; d++) {
    q_row[d] = Q[row * head_dim + d];   // VRAM -> registers, happens ONCE
}
```

#### Prefix caching

- If a request comes in, the system will assign the block to the request
- Number of assigned blocks = `ceil(prompt / block_size)`
- Request A : `[block A, block B, block C]`
- Block prefix hashing:
    - `BlockA_hash: Hash(BlockA)`
    - `BlockB_hash: Hash(BlockA_hash + BlockB)`
    - `BlockC_hash: Hash(BlockB_hash + BlockC)`
    - ...
- LRU eviction policy

```cpp
unordered_map<block_hash, KVCacheBlockNode*> mp;
class KVCacheBlockNode {
public:
    int block_id;
    string block_hash;
    int ref_cnt;
    KVCacheBlockNode* next;
    KVCacheBlockNode* prev;

    KVCacheBlockNode() {
        ref_cnt = 0;
    }
};

class KVCacheBlockDDL {
public:
    void append_tail(KVCacheBlockNode* node);
    KVCacheBlockNode* remove_head();
};
```

Only the node with `ref_cnt == 0` (not in use) will be stored in this DDL.

### LM cache

- if KV cache is evicted, it will go to other storage by hierarchy
- VRAM -> System RAM -> NVMe -> S3
- Use timing: recompute is expensive (big model or RAG (long prompt))
- Each tier keeps its own LRU DDL

### Continuous batching

- Request (Computed tokens, Total tokens that need to be computed)
- 2 queues (one waiting queue, one running queue)
- dynamic control (max request num & token budget)
- Chunked prefill and set a long prefill token threshold -> Split long tokens on prefill into small pieces to prevent a long prefill from preempting other decode

### Discrete Prefill & decode

- Prefill: Compute-intensive
- Decode: Memory-intensive
- Use a bridge to transfer KV cache between P/D
    - NVLink
    - RDMA
    - NXIL + TCP fallback

### Tensor parallelism

- Shard a model and put the model on multiple GPUs

## LLM-D (Routing)

### EPP (Endpoint picker)

#### Cache Affinity (x 0.4)

- Each pod will send out KV cache info by using ZMQ (in-memory) Publish

```
KV Event Config
{
    Event: BlockStored | BlockRemoved
    Publisher: "ZMQ",
    Endpoint: "tcp://ip:port",
    engineKeys: "Hash_by_vllm",
    Topic: "kv@<pod-ip>:<port>@model"
}
=>
Router will create Pod Entry map
- map<requestKey, vector<PodEntry>>
- map<EngineKey, RequestKey>
{
    engineKeys:  []BlockHash{e1, e2, e3},
    requestKeys: []BlockHash{r1, r2, r3},
    entries: []PodEntry{
        {PodIdentifier: "pod-A", DeviceTier: "gpu", Speculative: false},
        {PodIdentifier: "pod-A", DeviceTier: "gpu", Speculative: false},
        {PodIdentifier: "pod-A", DeviceTier: "gpu", Speculative: false},
}
```

- Speculative:
    - If two requests come in concurrently and cache is not calculated, two requests might go to different pods and do duplicate calculation.
    - Speculative will first store requestKeys and put NIL on engineKeys; after the pod finishes calculation, write back to podEntryMap
    - If one request sees NIL, it will wait for the other request to finish calculation
    - The NIL podEntry will be put in a TTL LRU cache. If it is not calculated in time, it will be removed.

#### Pod Select Filter

- cache affinity > 0.8 -> saturated?
    - workload = prefill throughput (token / s) x TTFT Penalty (Ms) = tokens to be dealt with
    - set a threshold; if workload is more than that, consider routing to a cold server because the hot server is too hot

- System workload (x 0.3)
- Network latency (x 0.2)
- Cache Availability (x 0.1)
