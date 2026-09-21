# AI Infra 学习进度

## 当前状态
- 阶段: 1/3 — llama.cpp 源码 × 大模型原理
- 位置: Week 1 · Day 2 ✅ 已完成
- 最近学习: 2026-09-21

## 已完成
- [x] W1D1: Transformer 架构总览 (prefill/decode、KV Cache 直觉、Decoder block 三件套)
- [x] W1D2: Self-Attention 细节 (QKV 投影、缩放、causal mask、MHA→MQA→GQA) + numpy 手写验证

## 下次计划
- [ ] W1D3: RoPE 旋转位置编码 (复数视角、相对性验证)

## 已解答疑问
- [x] 为什么 Q 要重算而 KV 可以缓存? → Q 是提问每次不同; K/V 只依赖自身输入, 建好不变
- [x] prefill 可并行 / decode 必须串行的原因 → 数据依赖: token i+1 依赖 token i 的输出

## 疑问清单
- [ ] RoPE 为什么能外推? (W1D3 解答)

## 实验记录
- 2026-09-21: numpy attention 与 torch.scaled_dot_product_attention 对照, 最大误差 8e-8; causal mask 上三角权重为 0 验证通过

## 作业
- [ ] W1D1: 手画 LLaMA block 数据流图 (不参考截图)