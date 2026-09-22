"""
W1D3 · RoPE 旋转位置编码 — numpy 手写 + 关键性质验证

核心公式:
    第 i 对 (维度 2i, 2i+1):  θ_i = 10000^(-2i/d_head)
    位置 m 的旋转: R(m·θ_i) = [[cos mθ_i, -sin mθ_i],
                              [sin mθ_i,  cos mθ_i]]
    RoPE(x, m) = R(m·θ) 作用于 x 的每一对

关键性质 (RoPE 的全部意义):
    <R(mθ)q, R(nθ)k>  =  <q, R((n-m)θ)k>
    → attention 打分只依赖相对位置 (m - n)
"""
import numpy as np

def precompute_freqs(d_head: int, positions: np.ndarray, theta_base: float = 10000.0) -> np.ndarray:
    """
    预计算每个位置 m、每对 i 的旋转角度 m·θ_i
    参数:
        d_head:    每个 head 的维度, 必须是偶数 (如 128)
        positions: 位置序列, shape (L,), 如 [0,1,2,...,L-1]
        theta_base: 基频, LLaMA 默认 10000
    返回:
        angles: shape (L, d_head//2), 每个位置、每对的角度 (弧度)
    """
    i = np.arange(d_head // 2)                                  # (d_head//2,) = (64,)
    theta_i = theta_base ** (-2 * i / d_head)                   # (64,)  每对的"基础转速"
    angles = np.outer(positions, theta_i)                       # (L, 64)  ← 关键广播
    return angles

def apply_rope(x: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    对 x 施加 RoPE
    参数:
        x:      shape (d_head,) 或 (L, d_head) —— 单个/多个 token 的 Q 或 K (一个 head 内)
        angles: shape (d_head//2,) 或 (L, d_head//2) —— 对应位置的旋转角度
    返回:
        x_rot:  旋转后的同 shape
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]                                    # (1, d_head)
    if angles.ndim == 1:
        angles = angles[np.newaxis, :]                          # (1, d_head//2)
    L = max(x.shape[0], angles.shape[0])                        # 位置数 (支持单 token 广播到多位置)
    d = x.shape[1]
    x = np.broadcast_to(x, (L, d)).copy()                       # 若 x 只有 1 个 token, 复制到 L 个
    angles = np.broadcast_to(angles, (L, d // 2)).copy()
    half = d // 2

    # 把连续的 2i, 2i+1 配对: x_even (L, half), x_odd (L, half)
    x_even = x[:, 0::2]                                         # 取维 0,2,4,...
    x_odd  = x[:, 1::2]                                         # 取维 1,3,5,...

    cos = np.cos(angles)
    sin = np.sin(angles)

    # 标准 RoPE 的"配对旋转"展开 (等价于复数 z·e^(jθ)):
    #   x'_{2i}   =  x_{2i}·cos θ_i  -  x_{2i+1}·sin θ_i
    #   x'_{2i+1} =  x_{2i}·sin θ_i  +  x_{2i+1}·cos θ_i
    x_even_new = x_even * cos - x_odd * sin
    x_odd_new  = x_even * sin + x_odd * cos

    out = np.empty_like(x)
    out[:, 0::2] = x_even_new
    out[:, 1::2] = x_odd_new
    return out.squeeze()


if __name__ == "__main__":
    np.random.seed(0)
    d_head = 128

    # ---------- 1. 形状检查 ----------
    q = np.random.randn(d_head)                                 # (128,)
    positions = np.array([0, 1, 5, 10])
    angles = precompute_freqs(d_head, positions)                # (4, 64)
    q_rot = apply_rope(q, angles)                               # (4, 128)
    print(f"q: {q.shape}  angles: {angles.shape}  q_rot: {q_rot.shape}")

    # 位置 0: 旋转角度全为 0, 输出应等于输入 (容差 1e-12)
    assert np.allclose(q_rot[0], q, atol=1e-12)
    print("✓ 位置 0 不旋转 (角度为 0)")

    # ---------- 2. 验证关键性质: <R(mθ)q, R(nθ)k> == <q, R((n-m)θ)k> ----------
    m, n = 3, 7
    angles_m = precompute_freqs(d_head, np.array([m]))          # (1, 64)
    angles_n = precompute_freqs(d_head, np.array([n]))          # (1, 64)
    angles_diff = precompute_freqs(d_head, np.array([n - m]))   # (1, 64)

    k = np.random.randn(d_head)                                 # (128,)

    lhs = apply_rope(q, angles_m) @ apply_rope(k, angles_n)     # <R(mθ)q, R(nθ)k>
    rhs = q @ apply_rope(k, angles_diff)                        # <q, R((n-m)θ)k>

    print(f"\n< R(mθ)q, R(nθ)k > = {lhs:.8f}")
    print(f"< q, R((n-m)θ)k >  = {rhs:.8f}")
    print(f"|lhs - rhs| = {abs(lhs - rhs):.2e}")
    assert abs(lhs - rhs) < 1e-9
    print("✓ 相对性恒等式成立: 打分只依赖相对位置 (n - m)")

    # ---------- 3. 直观理解: 旋转不改变向量长度 ----------
    orig_norm = np.linalg.norm(q)
    rot_norm = np.linalg.norm(q_rot[3])
    print(f"\n‖q‖ = {orig_norm:.6f}, ‖R(mθ)q‖ = {rot_norm:.6f}")
    assert np.isclose(orig_norm, rot_norm)
    print("✓ 旋转是正交变换, 不改变向量长度 (只改方向)")
