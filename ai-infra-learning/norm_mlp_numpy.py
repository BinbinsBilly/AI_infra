"""
W1D4 · RMSNorm 与 SwiGLU — numpy 手写验证

RMSNorm:  y = x / sqrt(mean(x^2) + eps) * gamma          (无减均值, 无 beta)
SwiGLU:   out = down_proj( silu(gate_proj(x)) * up_proj(x) )
          silu(x) = x * sigmoid(x)

验证点:
  1. RMSNorm 输出 shape 不变; gamma=1 时输出 RMS ≈ 1
  2. RMSNorm vs LayerNorm 的差异 = 均值未平移
  3. SwiGLU 全链路 shape 变化: (L,4096) → (L,11008) → (L,11008) → (L,4096)
  4. 参数量平衡: 3×(4096×11008) ≈ 2×(4096×16384)
"""
import numpy as np

# ---------- RMSNorm ----------
def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    x:     (L, d_model)
    gamma: (d_model,)
    返回:  (L, d_model)
    """
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)  # (L, 1) ← 沿特征维归约
    return x / rms * gamma                                       # 广播: (L,d)*(d,) → (L,d)

def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)                          # (L, 1)
    var = x.var(axis=-1, keepdims=True)                          # (L, 1)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta

def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))                                # x * sigmoid(x)

# ---------- SwiGLU MLP ----------
def swiglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray) -> np.ndarray:
    """
    x:      (L, d_model)  = (L, 4096)
    W_gate: (d_model, d_ff) = (4096, 11008)
    W_up:   (d_model, d_ff) = (4096, 11008)
    W_down: (d_ff, d_model) = (11008, 4096)
    返回:   (L, d_model)
    """
    g = x @ W_gate                       # (L, 11008)  gate 路投影
    u = x @ W_up                         # (L, 11008)  up 路投影
    h = silu(g) * u                      # (L, 11008)  逐元素门控
    out = h @ W_down                     # (L, 4096)   降回 d_model
    return out


if __name__ == "__main__":
    np.random.seed(0)
    L, d_model, d_ff = 3, 4096, 11008

    # ---------- 1. RMSNorm ----------
    x = np.random.randn(L, d_model) * 2.0 + 1.5      # 故意加偏移的输入
    gamma = np.random.rand(d_model) + 0.5            # (4096,)

    y = rmsnorm(x, gamma)
    print(f"[RMSNorm] x: {x.shape} → y: {y.shape}   (shape 不变 ✓)")
    assert y.shape == x.shape

    # gamma=1 时, 输出的 RMS 应 ≈ 1
    y_unit = rmsnorm(x, np.ones(d_model))
    rms_out = np.sqrt(np.mean(y_unit ** 2, axis=-1))
    print(f"[RMSNorm] gamma=1 时输出 RMS: {rms_out}  (应 ≈1)")
    assert np.allclose(rms_out, 1.0, atol=1e-3)

    # 与 LayerNorm 的差异: 均值不平移
    y_ln = layernorm(x, gamma, np.zeros(d_model))
    print(f"[对比] RMSNorm 输出均值: {y_unit.mean(axis=-1).round(3)}  (不为 0, 没做平移)")
    print(f"[对比] LayerNorm 输出均值: {y_ln.mean(axis=-1).round(6)}  (≈0, 做了平移)")

    # ---------- 2. SwiGLU ----------
    W_gate = (np.random.randn(d_model, d_ff) * 0.02).astype(np.float32)
    W_up   = (np.random.randn(d_model, d_ff) * 0.02).astype(np.float32)
    W_down = (np.random.randn(d_ff, d_model) * 0.02).astype(np.float32)

    xf = x.astype(np.float32)
    out = swiglu(xf, W_gate, W_up, W_down)
    print(f"\n[SwiGLU] x: {xf.shape}")
    print(f"[SwiGLU]   @W_gate → g: {(xf @ W_gate).shape}")
    print(f"[SwiGLU]   silu(g)*u:  {(silu(xf @ W_gate) * (xf @ W_up)).shape}")
    print(f"[SwiGLU]   @W_down → out: {out.shape}   (回到 d_model ✓)")
    assert out.shape == (L, d_model)

    # silu 性质: 可为小负值, 最小约 -0.278
    t = np.linspace(-10, 10, 2001)
    s = silu(t)
    print(f"\n[silu] 最小值 {s.min():.3f} (在 x={t[s.argmin()]:.2f} 处) — 注意不是硬截断到 0")

    # ---------- 3. 参数量平衡 ----------
    p_swiglu = 3 * d_model * d_ff
    p_std = 2 * d_model * (4 * d_model)
    print(f"\n[参数量] SwiGLU: 3×(4096×11008) = {p_swiglu/1e6:.1f}M")
    print(f"[参数量] 标准MLP: 2×(4096×16384) = {p_std/1e6:.1f}M")
    print(f"[参数量] 比值 = {p_swiglu/p_std:.3f}  (≈1, 这就是 11008 的由来: 8/3×4096→取整256倍数)")
