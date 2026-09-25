"""
W1D7 · Week 1 周检验: 从零拼一个 mini-LLaMA (2 层, GQA, 随机权重), 跑通 greedy 生成

架构 (LLaMA 家族, 玩具比例):
    token ids → embedding → N × [ RMSNorm → GQA-Attn(+RoPE) → 残差
                                  RMSNorm → SwiGLU      → 残差 ]
              → final RMSNorm → lm_head → logits → greedy

验证目标:
    1. 带 KV cache 的 decode 与不带 cache 的逐 token 全量重算, 输出序列完全一致
    2. FLOPs 对比: cache 让 decode 阶段投影 GEMM 计算量大幅下降
"""
import numpy as np

# ---------- 超参 (LLaMA-7B 等比例缩小) ----------
class Config:
    vocab_size = 320
    d_model    = 64          # LLaMA-7B: 4096
    n_heads    = 4           # Q 头数 (个数):  LLaMA-7B: 32
    n_kv_heads = 2           # KV 头数 (个数): LLaMA-3 8B: 8 (GQA)
    d_head     = d_model // n_heads   # 16 (每头维度, 与头数独立)
    n_layers   = 2
    d_ff       = 176         # ≈ 8/3 × d_model, 取 8 的倍数
    rope_theta = 10000.0
    eps        = 1e-5

# ---------- 基础算子 ----------
def rms_norm(x, gamma, eps):
    # x: (seq, d_model) → (seq, d_model);  公式: x / sqrt(mean(x²)+eps) * γ
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * gamma

def silu(x):
    return x / (1.0 + np.exp(-x))

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def rope(x, positions, theta):
    """
    旋转位置编码 (half-split 配对约定, 与 HF/llama.cpp 实现一致):
      配对 (i, i+d/2), 旋转角 m·θ_i,  θ_i = theta^(-2i/d)
    与 W1D3 讲的相邻配对 (2i, 2i+1) 数学等价, 仅维度排列不同
    x: (seq, n_heads, d_head) → 同 shape
    """
    seq, h, d = x.shape
    half = d // 2
    inv_freq = 1.0 / theta ** (np.arange(half) / half)      # (half,)
    ang = positions[:, None] * inv_freq[None, :]             # (seq, half)
    cos = np.cos(ang)[:, None, :]                            # (seq, 1, half)
    sin = np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x1 * sin + x2 * cos
    return out

# ---------- FLOPs 计数 (只统计 dense matmul, 即投影 GEMM) ----------
class FlopCounter:
    def __init__(self):
        self.matmul_flops = 0
    def matmul(self, a, b):
        # a: (m, k), b: (k, n) → (m, n), 计 2·m·k·n FLOPs
        self.matmul_flops += 2 * a.shape[0] * a.shape[1] * b.shape[1]
        return a @ b
FLOPS = FlopCounter()
TRACE = False   # 首次 forward 打印 shape 链

# ---------- KV Cache ----------
class KVCache:
    """每层一块连续内存: (max_seq, n_kv_heads, d_head); append 后返回全部历史"""
    def __init__(self, cfg, max_seq):
        self.k = [np.zeros((max_seq, cfg.n_kv_heads, cfg.d_head)) for _ in range(cfg.n_layers)]
        self.v = [np.zeros((max_seq, cfg.n_kv_heads, cfg.d_head)) for _ in range(cfg.n_layers)]
        self.n = [0] * cfg.n_layers
    def append(self, li, k, v):
        # k, v: (seq, n_kv_heads, d_head) → 返回 (total, n_kv_heads, d_head)
        s = k.shape[0]
        self.k[li][self.n[li]:self.n[li]+s] = k
        self.v[li][self.n[li]:self.n[li]+s] = v
        self.n[li] += s
        return self.k[li][:self.n[li]], self.v[li][:self.n[li]]

# ---------- 模型 ----------
class Layer:
    def __init__(self, cfg, rng):
        d, dkv = cfg.d_model, cfg.n_kv_heads * cfg.d_head
        def w(*shape, scale=0.02):
            return rng.normal(0, scale, shape)
        self.wq      = w(d, cfg.n_heads * cfg.d_head)   # (64, 64)  Q 满配
        self.wk      = w(d, dkv)                        # (64, 32)  ← GQA: K/V 更窄
        self.wv      = w(d, dkv)                        # (64, 32)
        self.wo      = w(cfg.n_heads * cfg.d_head, d)   # (64, 64)
        self.w_gate  = w(d, cfg.d_ff)                   # (64, 176)
        self.w_up    = w(d, cfg.d_ff)                   # (64, 176)
        self.w_down  = w(cfg.d_ff, d)                   # (176, 64)
        self.attn_norm = np.ones(d)                     # γ 初始化为 1
        self.mlp_norm  = np.ones(d)

def attention(x, L, cfg, positions, cache, li):
    """x: (seq, d_model), positions: (seq,) 绝对位置 → (seq, d_model)"""
    seq = x.shape[0]
    q = FLOPS.matmul(x, L.wq).reshape(seq, cfg.n_heads,    cfg.d_head)  # (seq, 4, 16)
    k = FLOPS.matmul(x, L.wk).reshape(seq, cfg.n_kv_heads, cfg.d_head)  # (seq, 2, 16)
    v = FLOPS.matmul(x, L.wv).reshape(seq, cfg.n_kv_heads, cfg.d_head)
    q = rope(q, positions, cfg.rope_theta)
    k = rope(k, positions, cfg.rope_theta)               # V 不旋转 [W1D3]
    if cache is not None:
        k_all, v_all = cache.append(li, k, v)            # (total, 2, 16)
    else:
        k_all, v_all = k, v
    total = k_all.shape[0]
    # GQA: 每 (n_heads/n_kv_heads)=2 个 Q 头共享一组 KV → repeat 到 4 头
    rep = cfg.n_heads // cfg.n_kv_heads
    k_all = np.repeat(k_all, rep, axis=1)                # (total, 4, 16) 计算时临时复制
    v_all = np.repeat(v_all, rep, axis=1)
    # 打分: (n_heads, seq, total)
    scores = np.einsum('shd,thd->hst', q, k_all) / np.sqrt(cfg.d_head)
    # causal mask (按绝对位置): query 只能看 position <= 自己的 key
    mask = positions[:, None] >= np.arange(total)[None, :]               # (seq, total)
    scores = np.where(mask[None], scores, -1e30)
    attn = softmax(scores)
    out = np.einsum('hst,thd->shd', attn, v_all).reshape(seq, -1)        # (seq, 64)
    if TRACE:
        print(f"    [trace] attn: q{q.shape} k/v{n_kv_shape}{k_all.shape} scores{scores.shape} → out{out.shape}")
    return FLOPS.matmul(out, L.wo)

n_kv_shape = "cache"  # 仅 trace 提示用

class MiniLLaMA:
    def __init__(self, cfg, seed=42):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.wte = rng.normal(0, 0.02, (cfg.vocab_size, cfg.d_model))    # (320, 64)
        self.layers = [Layer(cfg, rng) for _ in range(cfg.n_layers)]
        self.final_norm = np.ones(cfg.d_model)
        self.lm_head = rng.normal(0, 0.02, (cfg.d_model, cfg.vocab_size)) # (64, 320)

    def forward(self, token_ids, positions, cache=None):
        """token_ids: (seq,) → logits: (seq, vocab_size)"""
        cfg = self.cfg
        x = self.wte[np.asarray(token_ids)]              # (seq, 64) 查表
        if TRACE:
            print(f"    [trace] embed: {x.shape}")
        for li, L in enumerate(self.layers):
            if TRACE:
                print(f"    [trace] layer {li}:")
            h = rms_norm(x, L.attn_norm, cfg.eps)
            x = x + attention(h, L, cfg, positions, cache, li)           # 残差
            h = rms_norm(x, L.mlp_norm, cfg.eps)
            g = silu(FLOPS.matmul(h, L.w_gate))          # (seq, 176)
            u = FLOPS.matmul(h, L.w_up)                  # (seq, 176)
            x = x + FLOPS.matmul(g * u, L.w_down)        # 残差
            if TRACE:
                print(f"    [trace] mlp: gate/up{g.shape} → down out{ x.shape}")
        x = rms_norm(x, self.final_norm, cfg.eps)
        logits = FLOPS.matmul(x, self.lm_head)           # (seq, 320)
        return logits

# ---------- 生成 (greedy) ----------
def generate(model, prompt_ids, n_new, use_cache):
    cfg = model.cfg
    out = []
    if use_cache:
        cache = KVCache(cfg, len(prompt_ids) + n_new)
        pos = len(prompt_ids)
        logits = model.forward(prompt_ids, np.arange(pos), cache)   # ===== prefill =====
        nxt = int(logits[-1].argmax()); out.append(nxt)
        for _ in range(n_new - 1):                                  # ===== decode =====
            logits = model.forward([nxt], np.array([pos]), cache)   # 每步只进 1 个 token
            pos += 1
            nxt = int(logits[-1].argmax()); out.append(nxt)
    else:
        ids = list(prompt_ids)
        for _ in range(n_new):                                      # 每步全量重算
            logits = model.forward(ids, np.arange(len(ids)))
            nxt = int(logits[-1].argmax())
            out.append(nxt); ids.append(nxt)
    return out

if __name__ == "__main__":
    cfg = Config()
    model = MiniLLaMA(cfg, seed=42)
    prompt = [1, 5, 9, 13, 17]
    N_NEW = 24

    print("=" * 64)
    print("mini-LLaMA 配置 (LLaMA-7B 等比缩小)")
    print(f"  d_model={cfg.d_model}  n_heads={cfg.n_heads}  n_kv_heads={cfg.n_kv_heads}"
          f"  (GQA 组大小={cfg.n_heads//cfg.n_kv_heads})  d_head={cfg.d_head}")
    print(f"  n_layers={cfg.n_layers}  d_ff={cfg.d_ff}  vocab={cfg.vocab_size}")

    print("=" * 64)
    print("【0】首次 forward 的 shape 链 (prefill 5 tokens)")
    globals()['TRACE'] = True
    model.forward(prompt, np.arange(len(prompt)))
    globals()['TRACE'] = False

    print("=" * 64)
    print("【1】无 cache: 每步全量重算全部 token, greedy 生成")
    FLOPS.matmul_flops = 0
    out_nc = generate(model, prompt, N_NEW, use_cache=False)
    flops_nc = FLOPS.matmul_flops
    print(f"  生成 {N_NEW} tokens: {out_nc}")

    print("=" * 64)
    print("【2】带 KV cache: prefill 一次 + decode 每步只算 1 个新 token")
    FLOPS.matmul_flops = 0
    out_c = generate(model, prompt, N_NEW, use_cache=True)
    flops_c = FLOPS.matmul_flops
    print(f"  生成 {N_NEW} tokens: {out_c}")

    print("=" * 64)
    print("【3】一致性验证 (cache 是否改变结果)")
    print(f"  两条路径输出完全一致: {out_nc == out_c}")

    print("=" * 64)
    print("【4】投影 GEMM FLOPs 对比 (dense matmul, 不含 attention einsum)")
    print(f"  无 cache: {flops_nc/1e6:.2f} MFLOPs")
    print(f"  有 cache: {flops_c/1e6:.2f} MFLOPs")
    print(f"  计算量下降为 1/{flops_nc/flops_c:.1f}")

    print("=" * 64)
    print("【5】KV cache 显存 (fp32, 本玩具模型)")
    max_seq = len(prompt) + N_NEW
    per = cfg.n_layers * 2 * max_seq * cfg.n_kv_heads * cfg.d_head * 4
    print(f"  {max_seq} tokens: {per/1024:.1f} KB")
    print(f"  公式: 2 × n_layers × seq × n_kv_heads × d_head × 4B")
