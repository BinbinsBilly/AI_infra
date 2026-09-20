# 阶段 3: 手写 Toy 推理引擎 (6-8 周)

**目标**: 从零造一个迷你推理引擎 **mini-infer**, 支持加载真实 LLaMA 权重完成生成。路径: Python 原型 → C++ CPU → CUDA kernel → 批处理引擎 → 量化。每个 milestone 有明确验收标准, 全部通过即毕业。

## 工程纪律 (全程遵守)

- 每一步都与 golden reference (HF transformers 输出) 对齐后再前进
- git 小步提交, 一个功能一个 commit
- 关键算子写单测; 基准数据记录到 `ai-infra-learning/benchmarks.md`

## Milestone 0: 立项与环境 (2 天)

- 选定模型: 推荐 llama-3.2-1B-Instruct (或 TinyLlama-1.1B) 的 safetensors
- 用 transformers 跑通推理, 保存 5 个固定 prompt 的 logits/输出作为 golden
- 建立 mini-infer 仓库骨架: `python/` `c++/` `tests/` `benchmarks/`
- **验收**: golden 数据就位, 一键脚本可复现

## Milestone 1: NumPy 纯手写前向 (1 周)

- 读 safetensors: 手写解析 (header JSON + tensor 数据区), 不用第三方加载器
- 实现算子: embedding → RMSNorm → RoPE → GQA attention (带 KV cache) → SwiGLU → final norm → lm_head
- prefill 一次 + 逐 token decode; greedy 采样
- **验收**: 与 HF logits 最大误差 < 1e-3 (fp32); 能完整生成一段连贯文本

## Milestone 2: 引擎化 (1 周)

- 模块拆分: model / cache / sampler / generate 循环
- 多请求: 简单 batch (padding 对齐), 请求完成后退出 batch
- 采样: temperature / top-k / top-p + repetition penalty
- CLI: `python -m miniinfer -m <path> -p "..." --temp 0.7`
- **验收**: 两个并发请求 batch 推理, 同 seed 输出与单请求一致

## Milestone 3: C++ CPU 引擎 (2 周)

- mini-tensor 库 (仿 ggml 精简版):
  - arena 内存池 + tensor 结构 (data/type/shape/stride)
  - 算子: gemm/gemv、rmsnorm、rope、softmax、silu、mul、argmax
- 权重加载: safetensors 直读
- 串起前向; KV cache 用 ring buffer
- 优化: `-O3` + AVX2 gemm (分块/打包); 多线程 (可选)
- **验收**: 输出与 Milestone 2 一致 (同采样参数); 记录 decode 速度基线

## Milestone 4: CUDA Kernel (2 周)

- 环境准备: nvcc/nvidia-smi; nsys/ncu 使用方法
- kernel 清单 (从易到难):
  1. rmsnorm / rope / softmax 单算子 kernel
  2. gemv (权重矩阵按行并行, 保证合并访存)
  3. sgemm: naive → tiled (shared memory) → (可选) double buffering
  4. decode attention kernel: 单 query, 遍历 KV cache, 多 kv_head 并行
  5. prefill attention (flashattention 式 tiling, 至少分块 softmax)
- 内存: 权重与 KV 常驻显存; stream 双缓冲 (H2D 拷贝与计算重叠)
- 剖析: nsys 找 gap, ncu 看 kernel 指标 (带宽利用率/occupancy); 至少完成一轮"定位 → 优化 → 复测"
- **验收**: GPU 版输出对齐; 给出与 CPU 版的加速比和 kernel 耗时分解表

## Milestone 5: 批处理与 Paged KV (2 周)

- scheduler: 请求队列 + continuous batching (简化版: 到达即入 batch, 无抢占)
- paged KV cache: block 大小 16/32, block table, 空闲块分配器
- attention kernel 改造为按 block table 取 KV
- (可选) 用 python 或 cpp-httplib 套一个 OpenAI 兼容的 `/v1/chat/completions`
- 压测: 并发 (1/2/4/8) × 输出长度组合, 记录吞吐、TTFT、TPOT 曲线
- **验收**: 并发 8 的吞吐 ≥ 单请求吞吐 8 倍的 50% (证明 batching 生效); 曲线入 benchmarks.md

## Milestone 6: 量化与收官 (1 周)

- 权重量化: int8 per-channel 与 int4 group-wise (g=128) 两种
- 反量化 gemm 或 int kernel (任选其一实现到位)
- 评测: perplexity 变化 + 吞吐变化; 与 llama.cpp 量化结果横向对比
- 收官: 写 README (架构图、设计取舍、性能数据); 向导师做一次"技术答辩" (讲解 + 答疑)
- **验收**: 量化后 ppl 损失可接受 (以所选模型的 wikitext-2 基线为准); 全部 milestone 通过

## 毕业后延伸方向

- speculative decoding (ngram / 小模型 draft)
- CUDA graph 集成
- 多 GPU tensor parallel
- 给 llama.cpp / vLLM 提交一个小 PR (文档 / bugfix / kernel)

## 常见坑提示

- fp16 累加顺序不同导致的数值差异: 对齐验证用 rtol/atol, 不追求 bit-exact
- safetensors 的 dtype 与 shape 顺序要与 torch 期望核对
- RoPE 的 theta 与上下文缩放参数必须和 checkpoint 配置一致
- CUDA: 检查每个 API 返回值 (封装 CUDA_CHECK); 留意 bank conflict 与未合并访存
- KV cache 越界写入是 batch 实现的头号 bug 来源: 写批处理逻辑前先写单测
