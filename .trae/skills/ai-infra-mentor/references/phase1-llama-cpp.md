# 阶段 1: llama.cpp 源码 × 大模型原理 (4-6 周)

**目标**: 吃透 llama.cpp 的完整推理链路 — 从模型文件到 token 输出, 同时补齐 Transformer/LLM 原理。阶段结束后应能独立回答: **一个 token 是如何被生成的, 每一步发生在哪个文件哪个函数。**

**约定**: 以用户本地 llama.cpp 仓库为准。新版代码已拆分为 `src/llama-model.cpp`、`src/llama-kv-cache.cpp` 等模块; 若为老版本单文件 `src/llama.cpp`, 按对应逻辑阅读即可。

## Week 1: Transformer / LLM 原理

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | Decoder-only 架构总览 | token → embedding → N × (attn + mlp) → norm → logits; prefill vs decode 两阶段 | 画出 LLaMA block 数据流图 |
| 2 | Self-Attention | QKV 投影、缩放、causal mask; MHA → MQA → GQA 演进与显存收益 | numpy 手写 attention, 与 torch 对照 |
| 3 | RoPE | 旋转位置编码原理、复数视角、为何取代绝对位置编码 | numpy 实现 RoPE 并验证相对性 |
| 4 | RMSNorm 与 SwiGLU | 与 LayerNorm 对比 (为何去均值); gate/up/down 投影 | 手写 RMSNorm 与 SwiGLU |
| 5 | 词表与 tokenizer | BPE、SentencePiece (BPE/Unigram)、llama3 的 merged BPE 规则 | 用 tiktoken/transformers 观察 tokenize 结果 |
| 6 | 采样 | temperature、top-k、top-p、repetition penalty、greedy | 手写 top-k/top-p 采样器 |
| 7 | KV Cache 原理 | 为什么能省计算; 显存公式: 2 × layers × kv_heads × head_dim × ctx × dtype 字节 | 推导并计算 1B/7B 模型在 4k 上下文的 KV cache 大小 |

**周检验**: 用 numpy/PyTorch 从零拼出一个 2 层 mini-LLaMA 前向 (随机权重), 跑通 greedy 生成。

## Week 2: llama.cpp 上手 + ggml 张量库

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | 编译与运行 | cmake 构建; 下载一个小 GGUF 模型; llama-cli 关键参数 (`-m -p -n -t -ngl -c`) | 跑通并记录 tokens/s |
| 2 | 仓库结构 | `ggml/` (核心张量库) vs `src/` (llama 高层封装) vs `tools/` (CLI 工具); 后端目录 ggml-cuda/ggml-metal 等 | 画目录树并标注职责 |
| 3 | ggml 基础 | `ggml_tensor` 结构、`ggml_context`、内存池、tensor type (F16/F32/量化类型) | 读 `ggml.h`, 整理 tensor 结构字段表 |
| 4 | 计算图 | `ggml_mul_mat` 等算子只是"建图"不计算; `ggml_graph_compute_with_ctx` 触发执行; 拓扑排序 | 用纯 ggml API 写 `a @ b + relu` 小图并运行 |
| 5 | GGUF 格式 | header、kv metadata、tensor info 布局与对齐; mmap 加载 | 手写 python 脚本解析 GGUF 元数据 |
| 6 | backend 抽象 | `ggml_backend_buffer`、device、调度器如何把图分派到设备 | 理清一次 CPU buffer 分配的路径 |
| 7 | 机动 | 复习 + 答疑本周疑问清单 | — |

## Week 3: 模型加载链路

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | 入口与总控 | `tools/llama-cli/main.cpp` → `llama_init_from_gpt_params` → `llama_model_load` | gdb 断点跟一遍 |
| 2 | 超参与词表 | hparams 读取; `llama-vocab` 加载 BPE 词表与合并规则 | 打印 n_layer/n_head/n_kv_head 等 |
| 3 | tensor 映射 | `llama-arch` 的 tensor name 表; 权重名 → ggml tensor | 列出 LLaMA 一个 block 的全部 tensor 名 |
| 4 | 权重读取 | 逐 tensor 从 GGUF 读入 + 类型转换; mmap 共享物理页 | 观察一个权重的 shape/type |
| 5 | 显存/内存分配 | `ggml-alloc`; backend buffer 分配; `-ngl` 控制上卡层数 | 对比 `-ngl 0` 与 `-ngl 99` 的内存占用 |
| 6 | 加载完成态 | `model.layers[i]` 里有什么; 模型对象关系 | 画出加载后的对象关系图 |
| 7 | 周总结 | 不看笔记复述加载全流程 | — |

## Week 4: 前向计算图

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | decode 入口 | `llama_decode` → 图构建 (`llama-graph` / 老版 `build_llama_graph`) | 断点确认调用链 |
| 2 | embedding 查表 | `ggml_get_rows` | — |
| 3 | RMSNorm 算子 | `ggml_rms_norm` CPU 实现 (eps、多线程分块) | 阅读并注释关键行 |
| 4 | RoPE 算子 | rope 参数 (theta、freq_scale)、位置偏移 | 修改 theta 观察输出变化 |
| 5 | Attention 子图 | Q·K → mask → softmax → ·V; GQA 下 KV 的 broadcast (`ggml_repeat`) | 画出 attention 子图 |
| 6 | MLP 子图 | SwiGLU: silu(gate) × up → down 投影 | 与 Week1 手写版对照 |
| 7 | 输出头 | final norm → lm_head → logits; 为何只有最后一个 token 需要 logits | 解答并记录 |

## Week 5: KV Cache 与解码循环

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | KV cache 数据结构 | `llama-kv-cache`: slot、head、n_ctx 管理与扩容 | 打印 cache 状态 |
| 2 | 写入与读取 | `find_slot`、cache 满时的行为 | 构造超长输入触发扩容/失败 |
| 3 | 统一 KV 与 mask | GQA 共享 KV 的 mask 实现; `llama_kv_mask` 与 batch 的关系 | 理解 mask 如何作用于 KQ |
| 4 | logits 采样链路 | `llama-sampling`: dist、top-k/top-p/typical/temp 的实现顺序 | 用 `--logits` 打印观察分布 |
| 5 | 生成主循环 | llama-cli 的 decode → sample → 回填 token 循环 | 加 log 打印每步延迟, 区分 prefill/decode 耗时 |
| 6 | context 管理 | 上下文窗口满时的策略; 什么时候必须重算 | 实验超窗口行为 |
| 7 | 周总结 | 复述"一个 token 的旅程" (输入到输出全链路) | — |

## Week 6: 量化

| 天 | 主题 | 要点 | 动手 |
|---|---|---|---|
| 1 | 量化原理 | 线性量化、对称/非对称、per-channel vs per-block | 手推 Q4_0 还原公式 |
| 2 | 经典格式 | Q4_0/Q5_0/Q8_0 的 bit 布局 | python 实现 Q8_0 quantize/dequantize |
| 3 | K-quants | Q4_K/Q5_K/Q6_K: super-block、scales/mins 层级结构 | 解析 block 结构与大小 |
| 4 | 量化点积 | `vec_dot` 系列: dot(q,q) / dot(q,f); int8 计算路径 | 手写 Q4_0 dot (dequant 后验证) |
| 5 | 量化工具链 | `llama-quantize`; imatrix (重要性矩阵) 量化流程 | 同一模型做 Q4_K_M 与 Q4_0, 对比 ppl |
| 6 | 量化与性能 | 量化为何提速 (内存带宽瓶颈视角); 质量-速度取舍 | benchmark 对比三种量化 |
| 7 | 阶段总结 | 量化知识图谱; 回顾 week 1-6 疑问清单 | — |

## Week 7 (机动): 后端与进阶

| 天 | 主题 | 要点 |
|---|---|---|
| 1-2 | CPU 后端内幕 | threadpool 任务划分、SIMD (AVX2/AVX512/NEON)、ggml-cpu 算子分发 |
| 3-4 | CUDA 后端速览 | ggml-cuda 的算子注册与分派机制; 精读一个 kernel (如 `dequantize_mul_mat_vec`) |
| 5 | 阶段 1 终测 | 见下 |

## 阶段 1 验收 (终测)

通过标准: ≥80% 正确。

1. 画出 llama.cpp 从 GGUF 文件到首个 token 输出的完整调用链 (文件 + 函数名)
2. 写出 KV cache 显存公式并计算 7B 模型 (GQA) 在 4k 上下文下的占用
3. Q4_0 与 Q4_K 的区别; 为什么 K-quants 精度更好
4. ggml "建图与执行分离"的设计带来了什么好处
5. 手写 RoPE 公式; 给出 GQA 下 Q/K/V 的 shape
6. temperature 与 top-p 各自如何影响生成分布
7. prefill 和 decode 阶段的算子瓶颈为何不同 (计算 vs 带宽)
8. mmap 加载模型的好处
