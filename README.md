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

VLLM backend (Single node)
- PageAttention: a logical block table maps logical block to physical block
    - Block Calculation 
        - Each token: 2 (K & V) x num_layers (32) x num_kv_heads (8) x head_dim (128) x dtype_size (2) = 128 KB / token
        - Block Size: 128 x 16 (16 token per block)
        - Available KV cache memory = 90% x total memory - model_weights - activation memory
        - Num of block = Available KV cache memory ÷ Block Size
     
    - CUDA Kernel Optimization
        1.  logical block map to phyical block, calculate KV cache and write into phyical memory
        2.  GQA (Grouped Query Attention) 8 Q heads grouped to 2 KV heads
        3.  Softmax: Track the max value, scale the rest
            - naive: save all value then softmax, space complexity O(n)
            - Optimized: only save the max value, space complexity O(1)
        4. Partition long context and hand to multiple thread block to parallel process and then aggregate to softmax
        5. Prefill leverage FlashAttention kernel
            - FlashAttention:
                - Tilling: split QKV into small piece and load to SRAM to compute
                1. S = Q Kᵀ
                2. P = softmax(S)
                3. O = P V   
     
    - Prefix caching
        - If a request comes in, the system will assign the block to the request
        - Number of assign block = ceil(prompt / block_size)
        - Request A : [block A, block B, block C]
        - Block prefix hashing:
            - BlockA_hash: Hash(BlockA)
            - BlockB_hash: Hash(BlockA_hash + BlockB)
            - BlockC_hash: Hash(BlockB_hash + BlockC)
            - ...
        - LRU evict policy
        ```
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
        only the node with ref_cnt == 0 (not using) will store in this DDL. 

- LM cache
    - if KV cache be evicted, it will go to other storage by hierachy
    - VRAM -> System RAM -> NVMe -> S3
    - Use timing: recompute is expensive (Big model or RAG (long prompt))
    - Each tier keep their own LRU DDL
- Continuous batching
    - Request (Computed token, Total token need to be computed)
    - 2 queue (one waiting queue, one running queue)
    - dynamic control (max request num & token budget)
    - set a long prefill token threshold -> split long token on prefill to small piece to prevent a long prefill token preemptye other decode
- Discrete Prefill & decode
    - Prefill: Compute intensive
    - Decode: Memory intensive
    - Use Bridge to transfer KV cache between P/D
        - NVLink
        - RDMA
        - NXIL + TCP fallback 
- Tensor parrallelism
    - Shard a model and put model on multiple GPU

- LLM-D (Routing)
- EPP (Endpoint picker)
    - Cache Affinity (x 0.4)
        - Each pod will send out KV cache info by using ZMQ Publish
        ```
        KV Event Config
        {
            Publisher: "ZMQ",
            Endpoint: "tcp://ip:port",
            engineKeys: "Hash_by_vllm",
            requestKeys: "Hash_by_llm-d",
            Topic: "kv@<pod-ip>:<port>@model"
        }
        =>
        Router will create Pod Entry map
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
            - If two request come in concurrently and cache is not calculated, two requests might go to different pod and do duplicate calculation.
            - Speculative will first store requestKeys and put NIL on engineKeys, after the pod finish calculation write back to podEntryMap
            - If one request see NIL will wait for other request to finish calculation
            - The NIL podEntry will be put in a TTL LRU cache. If did not be calculated in time will be removed.
      - Pod Select Filter
          - cache affinity > 0.8 -> saturated?
              - workload = prefill throughput (token / s) x TTFT Penalty (Ms) = tokens to be deal with
              - set a threshold if workload more than that consider to route to cold server cuz hot server is too hot    
        
    - System workload (x 0.3)
    - Nework latency (x 0.2)
    - Cache Availability (x 0.1) 



