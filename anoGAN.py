import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import dataset_L2 as dataset


# 定义生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入 latent_dim 100, 输出尺寸为 7x7x1024
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 输出尺寸 14x14x512
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 输出尺寸 28x28x256
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 输出尺寸 56x56x128
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 输出尺寸 112x112x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 输出尺寸 224x224x3 (RGB图像)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # (64, 112, 112)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),  # (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),  # 输出尺寸 (1, 11, 11)
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, input):
        return self.main(input)


# 初始化生成器和判别器
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


def train_anogan(generator, discriminator, dataloader, num_epochs=20):
    real_label = 1
    fake_label = 0

    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(dataloader):
            inputs = data.cuda()

            # 训练判别器
            discriminator.zero_grad()
            outputs = discriminator(inputs)
            real_loss = criterion(outputs, torch.full_like(outputs, real_label, device=inputs.device))
            real_loss.backward()

            noise = torch.randn(inputs.size(0), 100, 1, 1, device=inputs.device)  # 随机噪声
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            fake_loss = criterion(outputs, torch.full_like(outputs, fake_label, device=inputs.device))
            fake_loss.backward()

            optimizer_d.step()

            # 训练生成器
            generator.zero_grad()
            outputs = discriminator(fake_images)
            gen_loss = criterion(outputs, torch.full_like(outputs, real_label, device=inputs.device))
            gen_loss.backward()

            optimizer_g.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'Loss D: {real_loss.item() + fake_loss.item():.4f}, Loss G: {gen_loss.item():.4f}')

    torch.save(generator.state_dict(), 'anogan_generator.pth')
    torch.save(discriminator.state_dict(), 'anogan_discriminator.pth')


def test_anogan(generator, test_loader):
    generator.eval()
    all_reconstructions = []
    all_inputs = []
    all_labels = []

    with torch.no_grad():
        for data, label in test_loader:
            data = data.cuda()

            # 使用随机噪声生成图像
            noise = torch.randn(data.size(0), 100, 1, 1, device=data.device)
            fake_images = generator(noise)

            # 保存输入和重构图像
            all_inputs.append(data.cpu().numpy())
            all_reconstructions.append(fake_images.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_inputs = np.concatenate(all_inputs)
    all_reconstructions = np.concatenate(all_reconstructions)
    all_labels = np.concatenate(all_labels)

    # 确保输入和重构图像的形状一致
    if all_inputs.shape[2:] != all_reconstructions.shape[2:]:
        # 调整重构图像的大小
        from torchvision.transforms import Resize
        resize = Resize(all_inputs.shape[2:])
        all_reconstructions = np.array([resize(torch.tensor(img)).numpy() for img in all_reconstructions])

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
is_train = False  # True-训练模型  False-测试模型
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
is_sampling = 'no_sampler'
# 数据准备
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train, is_sampling)

# 自编码器参数设置
input_dim = SIZE * SIZE * 3  # 根据图像尺寸调整


def load_weights(generator, discriminator):
    generator.load_state_dict(torch.load('anogan_generator.pth'))
    discriminator.load_state_dict(torch.load('anogan_discriminator.pth'))

if __name__ == '__main__':
    start = time.time()

    if is_train:

        # 训练自编码器
        train_anogan(generator, discriminator, train_loader, num_epochs=EPOCHS)
    else:
        # 加载权重
        load_weights(generator, discriminator)
        # 测试自编码器并进行异常检测
        test_anogan(generator, test_loader)

    print('Execution time:', time.time() - start)
