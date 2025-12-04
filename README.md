# 实验六：Inception v3 进行 FashionMNIST 分类

## 实验目的

使用 PyTorch 实现基于 Inception v3 预训练模型在 FashionMNIST 数据集上进行图像分类。

## 实验内容

### 1. 数据集与预处理
```python
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
train_full = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
test_full = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
n = 50
rng = np.random.default_rng(42)
train_idx = rng.choice(len(train_full), len(train_full)//n, replace=False)
test_idx = rng.choice(len(test_full), len(test_full)//n, replace=False)
train_loader = torch.utils.data.DataLoader(Subset(train_full, train_idx), batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(Subset(test_full, test_idx), batch_size=BATCH_SIZE, shuffle=False)
```

先定义transform用于给加载的数据进行预处理，使用Resize将图像尺寸调整到Inception模型所要求的输入尺寸，在将单通道的灰度图像转化为三通道的RGB图像，最后将数据转化为tensor。

加载完整的FashionMNIST的训练数据集以及测试数据集，对其进行上面定义的transform处理。定义n=50使用所有数据中1/50的数据，最后将数据转化为数据加载器dataloader。

### 2. 超参数设置
```python
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

定义batch_size作为数据加载器每次提取的数据量以及模型和数据训练的设备，即GPU

### 3. 模型结构
```python
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

加载inception v3模型并使用默认的预训练好的权重参数，也就是Pytorch上已经使用ImageNet数据集训练好的模型参数，使用这些参数来做迁移学习适配我们所要用的应用场景。仅修改fc层即最后输出的全连接层来匹配使用场景的输出要求，在这里是输入为fc层的输入特征数并输出10个类别。

将模型移到GPU上以加快训练速度，定义优化器为Adam优化器，学习率为1e-4.

### 4. 训练与评估流程
```python
EPOCHS = 20
accs, losses = [], []
with tqdm(total=EPOCHS, desc='Training Progress') as pbar:
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x).logits
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total_loss = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                total_loss += F.cross_entropy(out, y).item()
                correct += (out.argmax(dim=1) == y).sum().item()
        acc = correct / len(test_loader.dataset)
        avg_loss = total_loss / len(test_loader.dataset)
        accs.append(acc)
        losses.append(avg_loss)
        pbar.set_postfix({'Epoch': epoch + 1, 'Accuracy': acc, 'Loss': avg_loss})
        pbar.update(1)
```

设置Epoch数为20，使用tqdm作为训练进度条，相比于原先的每个Epoch输出损失和准确率更加直观简洁。每个epoch先将模型设置为训练模式，从训练数据加载器中取出数据移至GPU上，优化器梯度置零，得到模型输出的预测值，使用交叉熵损失函数计算损失并反向传播更新参数。

再将模型设置为评估模式，由于在评估模式下模型输出值为tensor没有logits值，所以无需加上.logits，将损失累加计算总损失值，再将预测准确的样本数量进行统计，每个epoch都计算平均损失和准确率并在训练过程中输出。

### 5. 可视化
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), accs, marker='o')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), losses, marker='o')
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()
```

使用在训练过程中得到的损失值以及准确率的列表，绘制出随epoch增加，loss值和acc值的变化曲线，通过可视化更加直观地看出训练的效果。

![alt text](images/output.png)

## 实验结果分析
- 使用预训练的 Inception v3 模型可以使用已经在大量数据下训练好的参数来匹配自己的实验环境要求，可以通过较少的数据和训练时间达到不错的效果。
- 通过修改预训练模型最后的输出全连接层可以将模型应用于自己的输出要求。
- 可以观察到准确率随训练逐步提升，损失逐步下降。

## 实验小结
本实验成功使用预训练的 Inception v3 模型在 FashionMNIST 数据集上进行迁移学习，实现了图像分类任务。通过替换输出层、调整输入数据数量和增加训练轮次，展示了迁移学习在小样本场景下的优势。实验验证了预训练模型可以有效地将在大规模数据集上学到的特征迁移到新任务上，大大提高了训练效率和模型性能，同时也加深了使用Inception模型进行训练的实践流程的理解。