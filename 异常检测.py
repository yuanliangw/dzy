import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import dataset_only_pos as dataset
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# 定义自编码器模型
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (16, 112, 112)
            nn.ReLU(True),
            nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (64, 56, 56)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (128, 28, 28)
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 输出尺寸: (256, 14, 14)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (128, 28, 28)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (64, 56, 56)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (16, 112, 112)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (3, 224, 224)
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, train_loader, num_epochs=10, lr=0.001):
    # criterion = nn.MSELoss()
    # 将 MSELoss 替换为 L1Loss
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, _ in train_loader:
            inputs = data.cuda()  # 使用CUDA
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

    torch.save(model.state_dict(), 'autoencoder_model.pth')
    print('Autoencoder model saved to autoencoder_model.pth')


def find_best_threshold(mse_per_sample, true_labels):
    fpr, tpr, thresholds = roc_curve(true_labels, mse_per_sample)
    auc_score = auc(fpr, tpr)
    print(f'AUC: {auc_score:.4f}')

    best_threshold = thresholds[0]
    best_f1 = 0.0

    for threshold in thresholds:
        predicted_labels = (mse_per_sample > threshold).astype(int)
        f1 = f1_score(true_labels, predicted_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f'Best threshold: {best_threshold:.4f}, Best F1-score: {best_f1:.4f}')
    return best_threshold, auc_score


def test_autoencoder(model, test_loader):
    model.eval()
    all_reconstructions = []
    all_inputs = []
    all_labels = []

    with torch.no_grad():
        for data, label in test_loader:
            data = data.cuda()

            inputs = data
            outputs = model(inputs)
            all_inputs.append(inputs.cpu().numpy())
            all_reconstructions.append(outputs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_inputs = np.concatenate(all_inputs)
    all_reconstructions = np.concatenate(all_reconstructions)
    all_labels = np.concatenate(all_labels)

    # 计算每个样本的均方误差
    mse_per_sample = np.mean((all_inputs - all_reconstructions) ** 2, axis=(1, 2, 3))

    true_labels = all_labels.astype(int)

    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(true_labels, mse_per_sample)
    auc_score = auc(fpr, tpr)
    print(f'AUC: {auc_score:.4f}')

    # 寻找最佳阈值 - 使用 F1 分数最大化
    best_threshold = thresholds[0]
    best_f1 = 0.0
    for threshold in thresholds:
        predicted_labels = (mse_per_sample > threshold).astype(int)
        f1 = f1_score(true_labels, predicted_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f'Best threshold: {best_threshold:.4f}, Best F1-score: {best_f1:.4f}')

    # 使用最佳阈值进行预测
    predicted_labels = (mse_per_sample > best_threshold).astype(int)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')


    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(mse_per_sample.reshape(-1, 1), cmap='YlGnBu', cbar=True, xticklabels=['MSE'], yticklabels=range(1, len(mse_per_sample) + 1))
    plt.title('Reconstruction Error Heatmap')
    plt.xlabel('MSE')
    plt.ylabel('Sample Index')
    plt.show()

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()


# 参数设置
PATH = 'data/labels_imbalance.csv'  # 训练数据路径
# TEST_PATH = 'data/exam_labels.csv' #
is_train = False # True-训练模型  False-测试模型
if is_train:
    TEST_PATH = ''
else:
    TEST_PATH = 'data/exam_labels.csv'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# 训练参数设置
SIZE = 224  # 图像进入网络的大小
BATCH_SIZE = 32  # batch_size数
EPOCHS = 60  # 迭代次数
sampling_way = {'over_sampler', 'down_sampler', 'no_sampler'}
is_sampling = 'down_sampler'
# 数据准备
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train, is_sampling)

# 自编码器参数设置
input_dim = SIZE * SIZE * 3  # 根据图像尺寸调整
if not is_train:
    autoencoder = ConvAutoencoder().cuda()
else:
    autoencoder = ConvAutoencoder().cuda()

if __name__ == '__main__':
    start = time.time()

    if is_train:
        # 训练自编码器
        train_autoencoder(autoencoder, train_loader, num_epochs=EPOCHS)
    else:
        # 测试自编码器并进行异常检测
        autoencoder.load_state_dict(torch.load('autoencoder_model.pth'))
        test_autoencoder(autoencoder, test_loader)

    print('Execution time:', time.time() - start)
