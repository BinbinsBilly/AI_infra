# 阶段 2: vLLM 源码带读 (4-6 周)

**目标**: 理解高吞吐 serving 引擎的设计 — 调度、显存管理、attention kernel、性能优化; 建立 llama.cpp (单请求友好) 与 vLLM (吞吐优先) 的对比视角。

**约定**: 以用户本地 vLLM 仓库为准; 以 v1 engine 为主线 (v0 已淘汰), 注意用户版本差异。前置: `pip install -e .` 装好源码版并能跑 offline inference 示例。

## Week 1: 架构与请求生命周期

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | 为什么需要 serving 引擎 | static/dynamic/continuous batching 对比; TTFT/TPOT/吞吐指标定义 | 手算三种 batching 的吞吐差异 |
| 2 | 仓库结构与入口 | `vllm/entrypoints` (cli/llm/api server)、engine、worker 分层 | 画分层架构图 |
| 3 | LLM 类 (offline) | `LLM.generate` → engine step 循环 | 跑通 examples/offline_inference |
| 4 | v1 架构 | CoreEngine (scheduler + executor)、processor、gpu_model_runner、async output 处理 | 画 v1 数据流图 |
| 5 | EngineCore 通信 | async 模式下的 zmq/messaging; detokenizer 独立进程 | 理解进程拓扑 |
| 6 | 一次 generate 的旅程 | request id → add_request → scheduler 入队 → step → RequestOutput | 打断点完整跟一遍 |
| 7 | 周总结 | 复述请求生命周期 | — |

## Week 2: Scheduler 与 Continuous Batching

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | 调度数据结构 | waiting/running 队列、RequestState 状态机 | 打印队列状态 |
| 2 | 调度决策 | token budget、KV block 余量、preempt 策略 | 构造打满场景观察 preemption |
| 3 | Chunked prefill | 为什么切分 prefill; `max_num_batched_tokens` 的作用 | 对比开关前后的 TTFT/TPOT |
| 4 | Prefix caching | block hash、自动前缀复用、evictor (LRU) | 相同前缀请求实验, 观察 cache hit |
| 5 | 调度公平性 | starvation 防止、请求优先级 | 思考: 为什么 prefill 优先会伤 decode 延迟 |
| 6 | 调度器演化 (选) | v0 与 v1 scheduler 差异, 为什么要重写 | — |
| 7 | 周总结 | 画出调度状态机 | — |

## Week 3: KV Cache 显存管理

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | 显存预算 | profile_run 推导 KV cache 可用空间; `gpu_memory_utilization` 语义 | 打印 num_gpu_blocks 并手算验证 |
| 2 | Block 结构 | KVCacheBlock、ref count、allocate/free 生命周期 | — |
| 3 | Block table | paged attention 的 block table, 类比 OS 虚拟内存 page table | 画图: 逻辑序列 → 物理块映射 |
| 4 | 写入路径 | forward 时 KV 写到哪; slot mapping 的计算 | 跟一次 forward 的 attn metadata |
| 5 | copy-on-write 与分支 | beam search / n>1 采样时的块共享 | 实验 n=3 采样观察块占用 |
| 6 | Prefix cache 实现 | hash 链、full/partial block 的处理 | — |
| 7 | 周总结 | 对比 llama.cpp 连续 KV 与 paged KV 的取舍 | — |

## Week 4: Model Runner 与 GPU 执行

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | batch 准备 | input_ids、positions、attn metadata 等 tensor 打包 | 断点看一个 decode batch 的形状 |
| 2 | 模型执行 | `vllm/model_executor/models/llama.py`: 前向与 HF 实现差异 (无 cache 对象、返回原始输出) | 对照 HF 代码阅读 |
| 3 | Attention 后端 | FLASHINFER / FLASH_ATTN 后端选择; prefill 与 decode 用不同 kernel | 打印实际使用的后端 |
| 4 | CUDA Graph | 为什么 decode 适合 graph replay; capture 的 padding 技巧 | 开/关 cuda graph 的性能对比 |
| 5 | 通信与 TP | `tensor_model_parallel_all_reduce`; 权重按 rank 切分方式 | 双卡跑 TP=2 观察切分 |
| 6 | worker/executor | 单机 vs Ray 分布式; 消息驱动的 step | — |
| 7 | 周总结 | 画出一次 decode step 的完整时间线 | — |

## Week 5: Attention Kernel 深读

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | PagedAttention kernel | block 级访存、thread block 到 (seq, head) 的并行划分 | 手画并行映射图 |
| 2 | decode kernel 优化 | split-K / 多 block 协作减少长序列尾部延迟 | 对比不同 ctx 长度下的表现 |
| 3 | FlashAttention 回顾 | online softmax、tiling 计算; prefill kernel 为什么天然适合 | 手推 online softmax |
| 4 | Metadata 构建 | slot_mapping、block_tables、query_start_loc 的生成与含义 | 打印真实值对齐理解 |
| 5 | Kernel 融合案例 | fused rmsnorm+quant、rotary embedding kernel | 阅读 flashinfer 调用侧 |
| 6 | Nsight 实践 | nsys profile 一次 serving, 读 timeline 找 gap | 找出 top 3 热点 |
| 7 | 周总结 | kernel 层知识图谱 | — |

## Week 6: 量化、采样与高级特性

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | 量化体系 | GPTQ/AWQ/FP8 权重加载路径; marlin kernel 的设计思想 | 跑两种量化对比吞吐/精度 |
| 2 | 采样实现 | sampler、penalties、logprobs; flashinfer 采样算子 | 实验 repetition penalty 效果 |
| 3 | Speculative decoding | draft-target 协作、接受率、为什么 decode 能快 | 跑 ngram/eagle 对比数据 |
| 4 | 并行策略 | TP/PP/EP/DP 适用场景与通信开销 | 画 TP=2 的通信图 |
| 5 | SGLang 对比阅读 | RadixAttention (radix tree 前缀复用)、调度哲学、与 vLLM 的分叉点 | 写对比笔记 |
| 6 | 生态与生产 | OpenAI 兼容 server、多模态、PD 分离 (disaggregated serving) 趋势 | — |
| 7 | 阶段 2 终测 | 见下 | — |

## 阶段 2 验收 (终测)

通过标准: ≥80% 正确。

1. 推导 continuous batching 相对 static batching 的吞吐差异
2. PagedAttention 解决了什么问题; block table 机制如何工作
3. 给定 `gpu_memory_utilization` 推导 KV block 数量
4. 为什么 decode 用 CUDA graph 而 prefill 不用
5. chunked prefill 对 TTFT 和 TPOT 分别有什么影响
6. prefix caching 的命中条件与典型收益场景
7. TP=2 时 llama 各层权重的切分方式与通信点位置
8. marlin kernel 为什么快 (访存视角)
9. speculative decoding 如何保证输出分布正确
10. vLLM vs SGLang vs llama.cpp: 各自设计目标与取舍
