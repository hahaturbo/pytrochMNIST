import torch
import torch.nn as nn
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import torchvision
import time
import matplotlib
import matplotlib.pyplot as plt

from model import CNN

start_time = time.time()
num_epochs = 30
batch_size = 64
learning_rate = 0.001


# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


# 从torchvision.datasets中加载一些常用数据集
train_dataset = normal_datasets.MNIST(
    root='./mnist/',  # 数据集保存路径
    train=True,  # 是否作为训练集
    transform=transforms.ToTensor(),  # 数据如何处理, 可以自己自定义
    download=True)  # 路径下没有的话, 可以下载

# 见数据加载器和batch
test_dataset = normal_datasets.MNIST(root='./mnist/',
                                     train=False,
                                     transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print(labels)
cv2.imshow('win', img)
key_pressed = cv2.waitKey(0)

cnn = CNN()
# 选择损失函数和优化方法
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# 若训练时测量值（如loss）停滞，则调整学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       patience=1,
                                                       verbose=1,
                                                       factor=0.5,
                                                       min_lr=1e-5)
# 使用gpu进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn.to(device)
loss_func.to(device)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch  {}/{}".format(epoch, num_epochs))
    print("-" * 10)
    for data in train_loader:
        X_train, y_train = data
        X_train, y_train = get_variable(X_train), get_variable(y_train)
        outputs = cnn(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = loss_func(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
    print("train ok ")
    testing_correct = 0
    for data in test_loader:
        X_test, y_test = data
        # 有GPU加下面这行，没有不用加
        X_test, y_test = X_test.cuda(), y_test.cuda()
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = cnn(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)
        # print(testing_correct)

    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(
        running_loss / len(train_dataset), 100 * running_correct / len(train_dataset),
        100 * testing_correct / len(test_dataset)))

# stop_time = time.time()
# print("time is %.3f" % (stop_time - start_time))

print("Finish!")
# 可视化loss
plt.plot(iteration_list, loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Loss vs Number of iteration")
plt.show()

# 可视化accuracy
plt.plot(iteration_list, accuracy_list, color="red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of iteration")
plt.show()

# Save the Trained Model
torch.save(cnn.state_dict(), 'model.pth')
