"""
W1D6 · 采样器手写验证

采样 pipeline:
    logits -> repetition penalty -> temperature -> softmax -> top-k -> top-p -> sample

各步骤公式:
    repetition:  l_i' = l_i / alpha   (若 token i 在历史中, alpha>1)
    temperature: p_i = exp(l_i/T) / sum_j exp(l_j/T)
    top-k:       只保留概率最高的 k 个, 其余置 0, 重新归一化
    top-p:       按概率从高到低累加, 保留累计概率 <= p 的最小集合, 重新归一化
"""
import numpy as np

def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """温度 softmax: p_i = exp(l_i/T) / sum exp(l_j/T)"""
    x = logits / T
    x = x - np.max(x)                          # 数值稳定
    e = np.exp(x)
    return e / e.sum()

def apply_repetition_penalty(logits: np.ndarray, history_ids: list[int], alpha: float = 1.1) -> np.ndarray:
    """对历史中出现过的 token 的 logit 除以 alpha (alpha>1 降低其概率)"""
    out = logits.copy()
    for tid in history_ids:
        out[tid] /= alpha
    return out

def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """只保留概率最高的 k 个, 其余置 0, 重新归一化"""
    if k <= 0:
        return probs
    topk_indices = np.argpartition(probs, -k)[-k:]      # 找到前 k 大的下标
    mask = np.zeros_like(probs)
    mask[topk_indices] = 1.0
    filtered = probs * mask
    s = filtered.sum()
    return filtered / s if s > 0 else filtered

def top_p_filter(probs: np.ndarray, p: float = 0.9) -> np.ndarray:
    """按概率从高到低累加, 保留累计概率 <= p 的最小集合 (nucleus sampling)"""
    sorted_idx = np.argsort(probs)[::-1]                # 从大到小排序的下标
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)                # 累计概率
    # 保留: 累计概率 <= p 的, 外加第一个超过 p 的 (保证覆盖 >= p)
    keep_mask = cumulative <= p
    keep_mask[np.argmax(~keep_mask)] = True             # 包含第一个使累计 > p 的
    keep_mask = np.roll(keep_mask, 0)
    keep_mask[0] = True                                 # 至少保留最高概率那个
    # 修正: 实际应保留累计<=p 的所有 + 第一个超出的
    keep_idx = sorted_idx[keep_mask]
    mask = np.zeros_like(probs)
    mask[keep_idx] = 1.0
    filtered = probs * mask
    s = filtered.sum()
    return filtered / s if s > 0 else filtered

def sample(probs: np.ndarray, greedy: bool = False) -> int:
    if greedy:
        return int(np.argmax(probs))
    return int(np.random.choice(len(probs), p=probs))

def full_pipeline(logits, T=1.0, top_k=0, top_p=1.0, rep_penalty=1.0, history=None, greedy=False):
    history = history or []
    if rep_penalty != 1.0:
        logits = apply_repetition_penalty(logits, history, rep_penalty)
    probs = softmax(logits, T)
    if top_k > 0:
        probs = top_k_filter(probs, top_k)
    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)
    return sample(probs, greedy=greedy), probs


if __name__ == "__main__":
    np.random.seed(42)
    vocab_size = 20
    # 模拟一批 logits (未归一化分数)
    logits = np.random.randn(vocab_size).astype(np.float32) * 2.0
    logits[3] = 5.0     # 让 token 3 概率最高
    logits[7] = 4.5
    logits[12] = 3.0

    print("=" * 60)
    print("【1. Greedy vs 随机采样】")
    p = softmax(logits, T=1.0)
    print(f"最高概率 token: {np.argmax(p)} (p={p.max():.3f})")
    print(f"greedy 选: {sample(p, greedy=True)}")
    print(f"随机采样 (T=1): {sample(p)}")

    print("\n" + "=" * 60)
    print("【2. Temperature 对比】")
    for T in [0.1, 0.7, 1.0, 2.0]:
        p = softmax(logits, T)
        top1 = np.argmax(p)
        print(f"T={T}: top1={top1} p={p[top1]:.3f}, 分布熵={-(p*np.log(p+1e-12)).sum():.3f}")

    print("\n" + "=" * 60)
    print("【3. Top-k (k=5)】")
    p = softmax(logits, 1.0)
    pk = top_k_filter(p, 5)
    print(f"原始前 5: {sorted(range(len(p)), key=lambda i:-p[i])[:5]}")
    print(f"top-k 后非零 token 数: {(pk > 0).sum()}")
    print(f"top-k 后概率和: {pk.sum():.6f} (应=1)")

    print("\n" + "=" * 60)
    print("【4. Top-p (p=0.9)】")
    pp = top_p_filter(p, 0.9)
    print(f"top-p 保留 token 数: {(pp > 0).sum()} (自适应)")
    print(f"top-p 后概率和: {pp.sum():.6f} (应=1)")

    print("\n" + "=" * 60)
    print("【5. Repetition Penalty (历史=[3], alpha=1.5)】")
    p_before = softmax(logits, 1.0)
    logits_pen = apply_repetition_penalty(logits, [3], 1.5)
    p_after = softmax(logits_pen, 1.0)
    print(f"token 3 概率: {p_before[3]:.4f} → {p_after[3]:.4f} (下降)")
    print(f"token 7 概率: {p_before[7]:.4f} → {p_after[7]:.4f} (不变)")

    print("\n" + "=" * 60)
    print("【6. 完整 pipeline 采样 10 次】")
    history = [3]
    for _ in range(10):
        tid, _ = full_pipeline(logits, T=0.8, top_k=10, top_p=0.95, rep_penalty=1.2, history=history)
        print(f"  抽到 token: {tid}")
