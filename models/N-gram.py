from collections import defaultdict, Counter
import random
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('punkt_tab')
import numpy as np
# 加载语料
corpus = nltk.corpus.gutenberg.raw('austen-emma.txt')[:10000]  # 用前 10000 字符
tokens = word_tokenize(corpus.lower())

# 构建 bigram 模型
bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
bigram_counts = Counter(bigrams)  # (w_{t-1}, w_t) 的计数
unigram_counts = Counter(tokens)  # w_{t-1} 的计数

# 计算概率（加一平滑）
V = len(unigram_counts)  # 词汇表大小
def bigram_prob(w1, w2):
    return (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)

# 生成文本
def generate_text(start_word, max_length=10):
    current = start_word
    result = [current]
    for _ in range(max_length - 1):
        # 基于概率采样下一个词
        candidates = [(w2, bigram_prob(current, w2)) for (w1, w2) in bigram_counts if w1 == current]
        if not candidates:
            break
        words, probs = zip(*candidates)
        current = random.choices(words, weights=probs, k=1)[0]
        result.append(current)
    return ' '.join(result)

# 测试
print("Generated text:", generate_text('emma'))

# 计算困惑度（简化版）
def perplexity(text):
    log_prob = 0
    n = len(text) - 1
    for i in range(n):
        w1, w2 = text[i], text[i+1]
        prob = bigram_prob(w1, w2)
        log_prob += -np.log2(prob) if prob > 0 else float('inf')
    return 2 ** (log_prob / n)

test_tokens = tokens[:100]
print("Perplexity:", perplexity(test_tokens))
