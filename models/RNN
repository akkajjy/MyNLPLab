import torch
import torch.nn as nn
import torch.optim as optim
import random

# 准备数据
text = """
The quick brown fox jumps over the lazy dog.
The cat runs fast and jumps high.
""".lower()
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

# 将文本转为索引序列
sequence = [char_to_idx[c] for c in text]
seq_length = 10  # 每次输入的序列长度

# 创建训练数据
def create_dataset(text, seq_length):
    inputs, targets = [], []
    for i in range(0, len(text) - seq_length):
        inputs.append([char_to_idx[c] for c in text[i:i+seq_length]])
        targets.append(char_to_idx[text[i+seq_length]])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

inputs, targets = create_dataset(text, seq_length)

# 定义 RNN 模型
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        x = self.embed(x)  # [batch, seq_length] -> [batch, seq_length, hidden_dim]
        out, hidden = self.rnn(x, hidden)  # out: [batch, seq_length, hidden_dim]
        out = self.fc(out)  # [batch, seq_length, vocab_size]
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim)

# 参数
hidden_dim = 20
n_layers = 1
model = CharRNN(vocab_size, hidden_dim, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
epochs = 100
batch_size = 32
for epoch in range(epochs):
    model.train()
    hidden = model.init_hidden(batch_size)
    total_loss = 0
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        if len(batch_inputs) == 0:
            break
        hidden = model.init_hidden(batch_inputs.shape[0])
        optimizer.zero_grad()
        hidden = hidden.detach()  # 防止梯度累积
        output, hidden = model(batch_inputs, hidden)
       # loss = criterion(output.view(-1, vocab_size), batch_targets)
        loss = criterion(output[:, -1, :].reshape(-1, vocab_size), batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss / (len(inputs) // batch_size)}")

# 生成文本
def generate_text(model, start_text, max_length=100):
    model.eval()
    chars = [char_to_idx[c] for c in start_text]
    hidden = model.init_hidden(1)
    result = list(start_text)
    with torch.no_grad():
        for _ in range(max_length - len(start_text)):
            input_tensor = torch.tensor([chars[-seq_length:]], dtype=torch.long)
            output, hidden = model(input_tensor, hidden)
            probs = torch.softmax(output[:, -1, :], dim=-1).squeeze()
            next_idx = torch.multinomial(probs, 1).item()
            result.append(idx_to_char[next_idx])
            chars.append(next_idx)
    return ''.join(result)

# 测试
print("Generated text:", generate_text(model, "the quick"))
