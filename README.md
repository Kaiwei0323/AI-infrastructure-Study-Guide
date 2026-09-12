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
| Constant memory | `__constant__ float x;` | Off-chip, cached on-chip | All threads, all blocks, host, read-only | Fast if cached | 64 KB |

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

<img width="994" height="1286" alt="image" src="https://github.com/user-attachments/assets/7fcf8541-8cad-48b6-9e16-3f7f5155f2f0" />

### Why we need self-attention?

Old model (RNN + Encoder / Decoder) drawback: Process tokens serially, and the old context will be forgotten and the new context will have more weight.

- Training: output can be processed in parallel
- Inference: output one at a time

### Encoding

`Input -> Embedding -> Multi-head attention`

- Input_embedding
    - Absolute PE: text embedding + text position embedding (`n x 512`)
    - RoPE: text embedding
        - q_rotated = R(position_q) @ q
        - k_rotated = R(position_k) @ k
        - score = q_rotated · k_rotated 
- Multi-head attention: `wQ + wK + wV` (`512 x 512`)

### Q, K, V

- Q: Ask a question?
- K: Yes or No
- V: Actual meaning

```
Standard SDPA (Scaled Dot-Product Attention):
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
Causal SDPA:
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + M) V
- M is a mask matrix added to attention scores that sets future position entries to -∞

Assume: E[q_i] = 0, E[k_i] = 0, Var(q_i) = 1, Var(k_i) = 1
Step 1 — Find E[q_i²] and E[k_i²]
    Var(q_i) = E[q_i²] − (E[q_i])²
    1        = E[q_i²] − 0²
    E[q_i²]  = 1
    E[k_i²]  = 1
Step 2 — Show E[q_i · k_i] = 0
    q_i and k_i are independent, so E[XY] = E[X]·E[Y]
    E[q_i·k_i] = E[q_i] · E[k_i] = 0 × 0 = 0
Step 3 — Find Var(q_i · k_i)
    Var(q_i·k_i) = E[(q_i·k_i)²] − (E[q_i·k_i])²
                 = E[q_i²·k_i²] − 0                (using Step 2)
                 = E[q_i²] · E[k_i²]                (independence)
                 = 1 × 1
                 = 1
Step 4 — Sum over all d_k dimensions
    q·k = q_1k_1 + q_2k_2 + ... + q_(d_k)k_(d_k)
    Since each term has variance 1, and all terms are independent,
    variances just add up:
    Var(q·k) = Var(q_1k_1) + Var(q_2k_2) + ... + Var(q_(d_k)k_(d_k))
             = 1 + 1 + ... + 1        (d_k terms total)
             = d_k
Step 5 — Scale by 1/√d_k to bring variance back to 1
    Var(q·k / √d_k) = (1/√d_k)² × Var(q·k)
                     = (1/d_k) × d_k
                     = 1
```

- Single head = Input_embedding (`n x 512`) * (`wQ` (`512 x 512`) + `wK` (`512 x 512`) + `wV` (`512 x 512`)) = `n x 512`, Attention(Q, K, V) = output
- Multi-head = Input_embedding (`n x 512`) * 8 * (`wQ` (`512 x 64`) + `wK` (`512 x 64`) + `wV` (`512 x 64`)) = `8 x (n x 64)` -> linear layer -> `n x 512`, Attention(Q, K, V) = output
    - Making `8 x (QKV)` is like feature extraction to make it more accurate

### MHA vs GQA vs MQA

| | Q heads | KV heads |
| --- | --- | --- |
| MHA | 8 | 8 |
| GQA (Group Query Attention) | 32 | 8 (Llama 3 8B) |
| MQA | many | 1 |

- Multiple Q heads can share one KV head: Q is "how I look up", K/V is the cached content. KV cache size uses `num_kv_heads`, not Q heads.

### Decoder

`output -> embedding -> Masked multi-head attention -> multi-head attention -> softmax -> output`


## Model Structure

- Safetensors: model weights
- Tokenizer: prompt -> token_ids
- 8B (model size): 16 GB (2x the parameters)
- MoE vs Non-MoE
    - MoE: EP (Expert Parallelism)
    - Non-MoE: TP (Tensor Parallelism)

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
    - `BlockA = [token1, token2, token3]`
    - `BlockA = [token4, token5, token6]`
    - `BlockA = [token7, token8, token9]`
    - `BlockA_hash: Hash(BlockA)`
    - `BlockB_hash: Hash(BlockA_hash + BlockB)`
    - `BlockC_hash: Hash(BlockB_hash + BlockC)`
    - ...
- LRU eviction policy
    - Free block (ref_cnt == 0) will be in double linkedlist
    - Used block will be in Request A : `[block A, block B, block C]`, block A.ref_cnt = 1, block B.ref_cnt = 1, block C.ref_cnt = 1, annd all three block will be removed from double linkedlist.
 
- Request
    - Token ids: [Token1, Token2, Token3, Token4, Token5, Token6]
    - Block Table: [Physical Block1, Physical Block2, Physical Block3]
    - Status: RUNNING | WAITING | FINISHED
    - seqlens_k = 6
    - seqlens_q = 2, if [Token1 - Tooken4] cached, only need to calculate [Token5, Token6]  

```cpp
unordered_map<block_hash, KVCacheBlockNode*> mp;
class KVCacheBlockNode {
public:
    int block_id;
    bytes SHA-256(block_hash);
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
- 2 queues (one waiting queue (prefill), one running queue (decode)) (deque)
- fix cap max_num_seqs = 256
- Chunked prefill and set a long prefill token threshold -> Split long tokens on prefill into small pieces to prevent a long prefill from preempting other decode

### varlen/packed attention
- Solve the padding add to prefill: padding will store in matrix and waste computing power
seq1: [t1 t2 t3 t4 t5]
seq2: [t1 t2 t3]
seq3: [t1 t2 t3 t4 t5 t6 t7 t8]
=>
seq1: [PAD PAD PAD t1 t2 t3 t4 t5]
seq2: [PAD PAD PAD PAD PAD t1 t2 t3]
seq3: [t1 t2 t3 t4 t5 t6 t7 t8]
Packed:
packed: [t1 t2 t3 t4 t5 | t1 t2 t3 | t1 t2 t3 t4 t5 t6 t7 t8]
              seq1            seq2                seq3
lengths:    [5, 3, 8]
cu_seqlens: [0, 5, 8, 16]
positions: [0 1 2 3 4 | 0 1 2 | 0 1 2 3 4 5 6 7] -> RoPE -> add to Q K vector

### Discrete Prefill & decode

- Prefill: Compute-intensive
    - FlashAttention
    - Chunked Prefill  
- Decode: Memory-intensive
    - PageAttention
    - Continue batching 
- Use a bridge to transfer KV cache between P/D
    - NCCL 
    - NVLink
    - RDMA
    - NXIL + TCP fallback

### Tensor parallelism

- Shard a model and put the model on multiple GPUs

## LLM-D (Routing)

### End-to-End Request Flow
```text
                         User / Client
                              │
                              │ HTTP / OpenAI API
                              ▼
                 ┌─────────────────────────┐
                 │   Inference Gateway     │
                 │         Envoy           │
                 │                         │
                 │ • Ingress               │
                 │ • TLS / Auth            │
                 │ • Request forwarding    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │         llm-d            │
                 │                          │
                 │      Router + EPP        │
                 │                          │
                 │ • Endpoint selection     │
                 │ • KV-cache awareness     │
                 │ • Load / token scoring   │
                 └────────────┬────────────┘
                              │
                    Request-level routing
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  Prefill Pool    │      │   Decode Pool    │
        │                  │      │                  │
        │   vLLM + EP      │      │    vLLM + EP     │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         ▲
                 │                         │
                 │       KV Cache          │
                 └────── Transfer ─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │   vLLM Runtime   │
        │                  │
        │   MoE Router     │
        └────────┬─────────┘
                 │
                 │ Token-level routing
                 │ Top-K Expert Selection
                 ▼
          ┌──────┼──────┐
          ▼      ▼      ▼
        GPU 0  GPU 2   GPU 3
       Expert 0 Expert 2 Expert 7
          │      │      │
          └──────┼──────┘
                 │
            EP / All-to-All
                 │
                 ▼
          GPU / Network
```

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
    - affinityMaxTTFTPenaltyMs=5000ms
    - XGBoost: Use regression model to predict pod workload

- System workload (x 0.3)
- Network latency (x 0.2)
- Cache Availability (x 0.1)

### Optimize CUDA Kernel
Element Wise
- Stride: BlockDim.x * GridDim.x - Single thread can process more than one operation within bound.
- Vectorize: float4* a1 = reinterpret_cast<float4*>(a); N / 4 - Single thread can process 4 data at once, reduce memory hit rate.
```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    int N4 = N / 4;

    for(size_t i = idx; i < N4; i += stride) {
        float4 AV = A4[i];
        float4 BV = B4[i];
        float4 CV;
        CV.x = AV.x + BV.x;
        CV.y = AV.y + BV.y;
        CV.z = AV.z + BV.z;
        CV.w = AV.w + BV.w;
        C4[i] = CV;
    }
    size_t remain = N4 * 4;
    for(size_t i = remain + idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
```

2D Matrix + Stride
```cpp
__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride_y = blockDim.y * gridDim.y;
    size_t stride_x = blockDim.x * gridDim.x;
    for(size_t r = row; r < N; r += stride_y) {
        for(size_t c = col; c < N; c += stride_x) {
            size_t idx = r * N + c;
            C[idx] = A[idx] + B[idx];
        }
    }
}
```

```cpp
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024

// 用 shuffle 取代原本的 shared-memory warp_reduction
__device__ float warp_reduce_shfl(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<unsigned int BlockSize>
__global__ void dot_product(const float* A, const float* B, float* result, int N) {
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    __shared__ float sdata[BlockSize];

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int N4 = N / 4;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    for (int i = idx; i < N4; i += stride) {
        float4 valA = A4[i];
        float4 valB = B4[i];
        sum += valA.x * valB.x + valA.y * valB.y + valA.z * valB.z + valA.w * valB.w;
    }

    int remain = N4 * 4;
    for (int i = remain + idx; i < N; i += stride) {
        sum += A[i] * B[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // shared memory 樹狀歸約，只做到剩下 32 個 (一個 warp) 為止
    if (BlockSize >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (BlockSize >= 512)  { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (BlockSize >= 256)  { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (BlockSize >= 128)  { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    // 最後 32 個執行緒改用 shuffle，不再碰 shared memory
    float val = 0.0f;
    if (tid < 32) {
        val = sdata[tid] + sdata[tid + 32];   // 先把 64 個元素合併成 32 個
        val = warp_reduce_shfl(val);
    }

    if (tid == 0) atomicAdd(result, val);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    cudaMemset(result, 0, sizeof(float));
    size_t GridSize = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dot_product<THREADS_PER_BLOCK><<<GridSize, THREADS_PER_BLOCK>>>(A, B, result, N);
}
```

Softmax
```cpp
#include <cuda_runtime.h>
#include <cfloat>

__device__ __forceinline__ void online_combine(float& m1, float& d1, float m2, float d2) {
    float new_m = fmaxf(m1, m2);
    d1 = d1 * expf(m1 - new_m) + d2 * expf(m2 - new_m);
    m1 = new_m;
}

__device__ void warp_shfl(float& m, float& d) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(mask, m, offset);
        float d2 = __shfl_down_sync(mask, d, offset);
        online_combine(m, d, m2, d2);
    }
}

template<unsigned int BlockSize>
__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float sdata_m[BlockSize];
    __shared__ float sdata_d[BlockSize];
    __shared__ float s_m;
    __shared__ float s_d;

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int N4 = N / 4;
    const float4* input4 = reinterpret_cast<const float4*>(input);

    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    for (int i = idx; i < N4; i += stride) {
        float4 val = input4[i];
        online_combine(local_m, local_d, val.x, 1.0f);
        online_combine(local_m, local_d, val.y, 1.0f);
        online_combine(local_m, local_d, val.z, 1.0f);
        online_combine(local_m, local_d, val.w, 1.0f);
    }

    int remain = N4 * 4;
    for (int i = remain + idx; i < N; i += stride) {
        online_combine(local_m, local_d, input[i], 1.0f);
    }

    sdata_m[tid] = local_m;
    sdata_d[tid] = local_d;
    __syncthreads();

    if (BlockSize >= 1024) { if (tid < 512) online_combine(sdata_m[tid], sdata_d[tid], sdata_m[tid + 512], sdata_d[tid + 512]); __syncthreads(); }
    if (BlockSize >= 512)  { if (tid < 256) online_combine(sdata_m[tid], sdata_d[tid], sdata_m[tid + 256], sdata_d[tid + 256]); __syncthreads(); }
    if (BlockSize >= 256)  { if (tid < 128) online_combine(sdata_m[tid], sdata_d[tid], sdata_m[tid + 128], sdata_d[tid + 128]); __syncthreads(); }
    if (BlockSize >= 128)  { if (tid < 64)  online_combine(sdata_m[tid], sdata_d[tid], sdata_m[tid + 64],  sdata_d[tid + 64]);  __syncthreads(); }

    if (tid < 32) {
        float m = sdata_m[tid];
        float d = sdata_d[tid];
        online_combine(m, d, sdata_m[tid + 32], sdata_d[tid + 32]);
        warp_shfl(m, d);
        if (tid == 0) {
            s_m = m;
            s_d = d;
        }
    }
    __syncthreads();

    for (int i = idx; i < N; i += stride) {
        output[i] = expf(input[i] - s_m) / s_d;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 1; // single block: reduction here is block-local, not cross-block

    softmax_kernel<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

Batch Norm VS Layer Norm VS RMS Norm VS Deep Norm
Batch Norm: Use batch or channel to do it but if batch is low or 1 perform bad but if batch is big like CNN perform well
        特征1   特征2   特征3   特征4
样本1    0.5     1.2    -0.3    2.1
样本2    0.8     0.9    -0.1    1.8
样本3    0.3     1.5    -0.5    2.3
样本4    0.6     1.1    -0.2    2.0
        ↓↓↓↓
       算这一列的
       均值和方差
    x̂ = (x − μ_batch) / √(σ²_batch + ε)
    y = γx̂ + β

Layer Norm: 
make μ = 0, σ² = 1, then use γ to scale and β to shift    
        特征1   特征2   特征3   特征4
样本1 → 0.5     1.2    -0.3    2.1   → 算这一行的均值方差
样本2 → 0.8     0.9    -0.1    1.8   → 算这一行的均值方差
样本3 → 0.3     1.5    -0.5    2.3   → 算这一行的均值方差

    x̂ = (x − μ_layer) / √(σ²_layer + ε)
    y = γx̂ + β

RMS Norm: trim the mean calculation on Layer Norm only divided by RMS (root-mean-square) and no β cuz will not force μ = 0
    x̂ = x / √((1/n)·Σxᵢ² + ε)
    y = γx̂

Deep Norm: 修复Post-LN在超深层数(1000+)下不稳定的问题
           做法:放大残差连接(乘α,α>1) + 缩小子层初始化(乘β)
           让梯度/更新幅度不管多深都能保持可控

    x_{l+1} = LN(α · x_l + G_l(x_l, θ_l))

    α  : 大于1的常数,根据深度(编码器N层/解码器M层)提前算好,不是学出来的
    G_l: 子层(注意力 或 前馈网络FFN)
    β  : 子层权重的初始化缩小因子(初始值比正常小)
    LN : 还是标准LayerNorm(跟前面Layer Norm那节公式一样)

    Post-LN:  x_{l+1} = LN(x_l + G_l(x_l))         → 效果好,但深了不稳定
    Pre-LN:   x_{l+1} = x_l + G_l(LN(x_l))         → 稳定,但"有效深度"变浅
    DeepNorm: Post-LN + α(放大残差) + β(缩小初始化) → 效果接近Post-LN,稳定性接近甚至超过Pre-LN

Why make 𝛼 (residual) BIGGER
- the old, already-accumulated signal should dominate; each new layer's fresh contribution should only be a small addition relative to it.
Why make β (weight init) SMALLER
- you're making start out quiet/small — so even before you multiply the residual by α, the "new contribution" isn't overwhelming to begin with.
Total signal magnitude must stay bounded as you stack many layers.

### Profiling
Nsight System
- Profile the whole system timeline from CPU, CUDA, API, cudaMemcpy, Kernel launch, GPU idle gap, overlap
Nsight compute
- Microscope on one kernel from occupancy, DRAM %, warp stalls, registers, source lines (map GPU performance to each line, see which line cause memory traffic / stall / instruction), roofline (memory bound or compute bound, try to push from memory bound (GPU still got idle computing power to compute bound (might need to upgrade hardware)))

### Tensor Parallel
- Column Parallel / Row Parallel: Shard weight, QKV, Token Embedding across GPUs, replica: RoPE, RMS, Layer Norms.
- Column Parallel:
```
X = [1, 2, 3, 4] (1, 4)

W0 (GPU0)              W1 (GPU1)
| 1   2 |              | 3   4 |
| 5   6 |              | 7   8 |
| 9  10 |              |11  12 |
|13  14 |              |15  16 |
(4, 2)                 (4, 2)
GPU0:  Y0 = [1,2,3,4] @ W0 = [90, 100]
GPU1:  Y1 = [1,2,3,4] @ W1 = [110, 120]
output: [90, 100, 110, 120]
```
- Row Parallel
```
X0 = [1, 2]            X1 = [3, 4]

W0 (GPU0)              W1 (GPU1)
| 1   2   3   4 |      | 9  10  11  12 |
| 5   6   7   8 |      |13  14  15  16 |
(2, 4)                 (2, 4)

GPU0:  Y0 = [1, 2] @ W0 = [11, 14, 17, 20]
GPU1:  Y1 = [3, 4] @ W1 = [79, 86, 93, 100]

all_reduce SUM:
  [11, 14, 17,  20]
+ [79, 86, 93, 100]
= [90, 100, 110, 120]
```

Token -> Embedding -> [Norm -> Attention -> + Residual -> Norm -> MLP -> + Residual] x N -> Final Norm -> LM Head -> Logits -> softmax -> sampling (pick index by proability with temperature) -> output (next token)
MLP: X -> Column Parallel -> SwiGLU (SiluAndMul) -> Row Parallel -> All-Reduce -> Output

