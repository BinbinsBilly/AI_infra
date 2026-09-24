"""
W1D5 · Tokenizer 观察
用 tiktoken (GPT-4 的 cl100k_base, 与 LLaMA-3 同思路的 BPE) 观察切分行为
"""
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

print(f"词表大小: {enc.n_vocab}\n")

samples = [
    "Hello, world!",
    "unfriendliness",           # 生僻词 → 拆成子词
    "The quick brown fox jumps over the lazy dog.",
    "大模型推理引擎",            # 中文: 字节级 fallback + 子词
    "1234567890",               # 数字切分
    "def hello(): return 42",   # 代码
    "😀🎉",                      # emoji: 多字节
]

for text in samples:
    ids = enc.encode(text)
    tokens = [enc.decode([i]) for i in ids]
    print(f"文本:   {text}")
    print(f"token数: {len(ids)}")
    print(f"tokens: {tokens}")
    print(f"ids:    {ids}\n")

# 观察单个字符 vs 合并后的词表占比
print("=" * 60)
print("观察: 词表里有多少 token 长度 >= 3 (合并出来的子词)?")
single_char = multi_char = 0
for i in range(enc.n_vocab):
    try:
        t = enc.decode([i])
    except Exception:
        continue
    if len(t) == 1:
        single_char += 1
    else:
        multi_char += 1
print(f"单字符 token: {single_char}")
print(f"多字符 token (合并产物): {multi_char}")
print(f"占比: {multi_char/(single_char+multi_char)*100:.1f}%")
