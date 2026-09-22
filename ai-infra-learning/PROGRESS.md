# AI Infra 学习进度

## 当前状态
- 阶段: 1/3 — llama.cpp 源码 × 大模型原理
- 位置: Week 1 · Day 3 ✅ 已完成
- 最近学习: 2026-09-22

## 已完成
- [x] W1D1: Transformer 架构总览 (prefill/decode、KV Cache 直觉、Decoder block 三件套)
- [x] W1D2: Self-Attention 细节 (QKV 投影、缩放、causal mask、MHA→MQA→GQA) + numpy 手写验证
- [x] W1D2 追加答疑: 多头切分方式、KV cache 显存公式、n_heads vs d_head、GQA 下 K/V 总维度
- [x] W1D3: RoPE 旋转位置编码 (配对旋转、复数视角、相对性恒等式、外推性与 NTK/YaRN) + numpy 手写验证

## 下次计划
- [ ] W1D4: RMSNorm 与 SwiGLU (为何去均值、gate/up/down 投影形状) — 全程带 shape 标注 + 矩阵图

## 已解答疑问
- [x] 为什么 Q 要重算而 KV 可以缓存? → Q 是提问每次不同; K/V 只依赖自身输入, 建好不变
- [x] prefill 可并行 / decode 必须串行的原因 → 数据依赖: token i+1 依赖 token i 的输出
- [x] d_head 是什么? → 每个 head 的维度 = d_model/n_heads, 与 n_heads (头的个数) 是两个独立概念
- [x] GQA/MQA 中 K/V 的 dim? → 每个 KV head 维度仍为 d_head; K/V 总维度 = n_kv_heads × d_head (GQA 下 < d_model)
- [x] RoPE 为什么能外推? → 相对性使距离 d 可泛化; 但高频对周期性折叠导致实际外推掉点, 需 NTK/YaRN scaling

## 疑问清单
- [ ] RoPE 为什么能外推? (W1D3 解答)

## 教学偏好 (导师必读)
- **矩阵维度优先**: 讲张量运算必须写出 shape 变化链, 个数与维度严格区分, 尽量画矩阵图/数据流图, 用具体数字 (LLaMA-7B: d_model=4096, h=32, d_head=128) 举例

## 实验记录
- 2026-09-21: numpy attention 与 torch.scaled_dot_product_attention 对照, 最大误差 8e-8; causal mask 上三角权重为 0 验证通过 (attention_numpy.py)
- 2026-09-22: numpy RoPE 验证: 位置0不旋转、相对性恒等式 |lhs-rhs|=0、旋转保长 (rope_numpy.py)

## 作业
- [ ] W1D1: 手画 LLaMA block 数据流图 (不参考截图)
- [ ] 自查: 默写 KV cache 显存公式, 并手算 LLaMA-3 8B (GQA, n_layers=32, n_heads=32, n_kv=8, d_head=128, fp16, 4k ctx) 的 KV cache 大小 → 答案应为 512 MB