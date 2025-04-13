import numpy as np
from collections import defaultdict

# 假设词汇表和数据
vocab = ["cat", "dog", "run", "jump"]  # 简化示例
word_to_idx = {w: i for i, w in enumerate(vocab)}
V, d = len(vocab), 2  # 词汇表大小，向量维度

# 初始化词向量
W = np.random.randn(V, d) * 0.01  # 目标词向量
W_context = np.random.randn(V, d) * 0.01  # 上下文向量

# 模拟训练数据：(目标词索引, 上下文词索引)
training_pairs = [(word_to_idx["cat"], word_to_idx["dog"]),
                  (word_to_idx["dog"], word_to_idx["run"])]

# 训练 Skip-gram
learning_rate = 0.01
for epoch in range(100):
    loss = 0
    for target, context in training_pairs:
        # 前向传播
        h = W[target]  # 目标词向量
        u = np.dot(W_context, h)  # 得分
        y_pred = 1 / (1 + np.exp(-u[context]))  # sigmoid
        
        # 损失（简化，只考虑正样本）
        loss += -np.log(y_pred)
        
        # 反向传播（更新向量）
        grad = y_pred - 1
        W_context[context] -= learning_rate * grad * h
        W[target] -= learning_rate * grad * W_context[context]
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# 评估：计算词向量相似度
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Similarity(cat, dog):", cosine_similarity(W[word_to_idx["dog"]], W[word_to_idx["cat"]]))
