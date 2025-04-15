import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random

# 准备数据
corpus = """
The cat is running. The dog is barking. A cat jumps high.
The dog runs fast. Cats and dogs play together.
""".lower().split()
vocab = list(set(corpus))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# 生成训练对（窗口大小=2）
window_size = 2
training_data = []
for i in range(window_size, len(corpus)):
    context = [word_to_idx[corpus[i-j]] for j in range(window_size, 0, -1)]
    target = word_to_idx[corpus[i]]
    training_data.append((context, target))

# 定义模型
class NeuralLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, window_size):
        super(NeuralLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(window_size * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)  # [batch, window_size, embed_dim]
        x = x.view(x.size(0), -1)  # [batch, window_size * embed_dim]
        x = self.fc1(x)  # [batch, hidden_dim]
        x = self.relu(x)
        x = self.fc2(x)  # [batch, vocab_size]
        return x

# 参数
vocab_size = len(vocab)
embed_dim = 10
hidden_dim = 20
model = NeuralLM(vocab_size, embed_dim, hidden_dim, window_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for context, target in training_data:
        context = torch.tensor(context, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        optimizer.zero_grad()
        output = model(context.unsqueeze(0))  # Add batch dimension
        loss = criterion(output, target.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(training_data)}")

# 生成文本
def generate_text(model, start_context, max_length=10):
    model.eval()
    context = [word_to_idx[w] for w in start_context]
    result = start_context[:]
    with torch.no_grad():
        for _ in range(max_length - len(start_context)):
            context_tensor = torch.tensor(context[-window_size:], dtype=torch.long)
            output = model(context_tensor.unsqueeze(0))
            probs = torch.softmax(output, dim=1).squeeze()
            next_idx = torch.multinomial(probs, 1).item()
            result.append(idx_to_word[next_idx])
            context.append(next_idx)
    return ' '.join(result)

# 测试
start_context = ['the', 'cat']
print("Generated text:", generate_text(model, start_context))
