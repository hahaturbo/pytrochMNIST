import torch.nn as nn


# 两层卷积
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 使用序列工具快速构建
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 卷积层
        # i --> input channels
        # 6 --> output channels
        # 5 --> kernel size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 全连接层
        # 16 * 4 * 4 --> input vector dimensions
        # 120 --> output vector dimensions
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 --> ReLu --> 池化
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), (2, 2))
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), (2, 2))

        # reshape, '-1'表示自适应
        # x = (n * 16 * 4 * 4) --> n : input channels
        # x.size()[0] == n --> input channels
        x = x.view(x.size()[0], -1)

        # 全连接层
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x
