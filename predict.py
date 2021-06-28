import torchvision as tv
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import Net


def get_dataset():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 测试集
    testset = tv.datasets.MNIST(
        root='./mnist/',
        train=False,
        download=True,
        transform=transform,
    )
    testloadter = DataLoader(
        dataset=testset,
        batch_size=4,
        shuffle=True,
    )

    return len(testset), testloadter


def main():
    # 数字类别
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # 初始化网络参数
    model = Net()

    # 加载模型
    model.load_state_dict(torch.load('model.pth'))

    if torch.cuda.is_available():
        # 使用GPU
        model.cuda()

    # get dataloader
    data_len, dataloader = get_dataset()

    # 预测正确的个数
    correct_num = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        # 取最大值为预测结果
        _, predicted = torch.max(outputs, 1)

        for j in range(len(predicted)):
            predicted_num = predicted[j].item()
            label_num = labels[j].item()
            # 预测值与标签值进行比较
            if predicted_num == label_num:
                correct_num += 1

    # 计算预测准确率
    correct_rate = correct_num / data_len
    print('correct rate is {:.3f}%'.format(correct_rate * 100))


if __name__ == "__main__":
    main()
