"""
W1D2 动手: numpy 手写 scaled dot-product attention, 与 PyTorch 对照
运行: python attention_numpy.py
"""
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # 减最大值防溢出
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_numpy(Q, K, V, mask=None):
    """
    Q, K, V: shape (seq_len, d_k)
    mask: shape (seq_len, seq_len), 需要被 mask 的位置填 -inf
    返回: (seq_len, d_v), (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    # 缩放点积
    scores = Q @ K.T / np.sqrt(d_k)          # (seq, seq)
    if mask is not None:
        scores = scores + mask               # 上三角 -inf
    weights = softmax_np(scores, axis=-1)    # 每行和为1
    output = weights @ V                     # (seq, d_v)
    return output, weights


def causal_mask(seq_len: int) -> np.ndarray:
    # 上三角(i<j)填 -inf, 其余 0
    mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
    mask = np.triu(mask, k=1)                # 保留主对角线及以下
    return mask


def main():
    np.random.seed(42)
    seq_len, d_k, d_v = 5, 8, 8

    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_v).astype(np.float32)
    mask = causal_mask(seq_len)

    out_np, w_np = attention_numpy(Q, K, V, mask)

    print("=== numpy attention ===")
    print("weights (每行和应为1):")
    print(np.round(w_np, 4))
    print("row sums:", np.round(w_np.sum(axis=-1), 6))
    print()

    if HAS_TORCH:
        Qt = torch.from_numpy(Q)
        Kt = torch.from_numpy(K)
        Vt = torch.from_numpy(V)
        # torch 的 causal mask: True 表示要被 mask 掉
        is_causal = True
        out_torch = torch.nn.functional.scaled_dot_product_attention(
            Qt.unsqueeze(0), Kt.unsqueeze(0), Vt.unsqueeze(0),
            is_causal=is_causal
        ).squeeze(0).numpy()
        print("=== 与 torch 对照 ===")
        print("max abs diff:", np.max(np.abs(out_np - out_torch)))
        assert np.allclose(out_np, out_torch, atol=1e-5), "数值不一致!"
        print("PASS: numpy 实现与 torch 一致")
    else:
        print("(未安装 torch, 跳过对照)")

    # 思考题: 验证 mask 生效——weights 的上三角应为 0
    print()
    print("上三角权重应为 0 (i<j):")
    upper = w_np[np.triu_indices(seq_len, k=1)]
    print("max upper-tri weight:", np.max(upper))
    assert np.allclose(upper, 0), "mask 未生效!"
    print("PASS: causal mask 生效")


if __name__ == "__main__":
    main()
