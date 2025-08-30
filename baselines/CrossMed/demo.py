import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 假设数据集：药物集合和历史健康评分
# 药物用编号表示，比如药物1, 2, 3，健康评分是一个整数
data = [
    {'药物': [1, 2, 3], '健康评分': 65},  # 第一次记录
    {'药物': [2, 3], '健康评分': 70},  # 第二次记录
    {'药物': [1, 3], '健康评分': 60},  # 第三次记录
    {'药物': [2], '健康评分': 75},  # 第四次记录
    {'药物': [1, 3], '健康评分': 68},  # 第五次记录
    {'药物': [3], '健康评分': 72}  # 第六次记录
]

# 数据准备
# 1. 将药物的集合表示成one-hot编码或嵌入向量，健康评分则归一化处理
medication_vocab_size = 4  # 假设有4种药物（药物编号1, 2, 3）
max_medications = 3  # 每次最多有3种药物


# 创建one-hot编码的药物特征
def encode_medications(medications, vocab_size):
    one_hot = np.zeros((max_medications, vocab_size))
    for i, med in enumerate(medications):
        one_hot[i, med - 1] = 1
    return one_hot


# 准备输入数据和目标数据
X = []
y = []

for entry in data:
    meds_encoded = encode_medications(entry['药物'], medication_vocab_size)
    X.append(meds_encoded)  # 输入是one-hot编码的药物
    y.append(entry['健康评分'])  # 输出是健康评分

X = np.array(X)
y = np.array(y)

# 归一化健康评分（如果需要，可以跳过这步）
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y_scaled)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# 2. 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size)  # 初始化隐藏状态
        out, _ = self.gru(x, h_0)  # GRU 前向传播
        out = self.fc(out[:, -1, :])  # 通过全连接层得到输出
        return out


# 超参数
input_size = medication_vocab_size  # 药物种类的数量
hidden_size = 64  # GRU隐藏层的大小
output_size = 1  # 预测一个健康评分值

# 3. 初始化模型、损失函数和优化器
model = GRUModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# 6. 反归一化预测值
predicted_scores = scaler_y.inverse_transform(predictions.detach().numpy())
actual_scores = scaler_y.inverse_transform(y_test.detach().numpy())

# 输出结果
print(f'Predicted Health Scores: {predicted_scores.flatten()}')
print(f'Actual Health Scores: {actual_scores.flatten()}')