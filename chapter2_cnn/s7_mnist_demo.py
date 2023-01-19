from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import accuracy_score
from data_set import filepaths as fp


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10))

def evaluate_accuracy(net, data_iter):
    net.eval()  # 设置为评估模式
    total_acc = 0
    count = 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat = torch.argmax(y_hat,dim=1)
            total_acc += accuracy_score(y.cpu(), y_hat.cpu())
            count+=1
    return total_acc / count

def load_data(isTrain=True,batch_size=256,shuffle=True):
    data = torchvision.datasets.MNIST(
        root=fp.MNIST,
        train=isTrain,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def train(epochs=10, lr=0.5,batch_size=256):
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    train_load = load_data(True,batch_size,True)
    test_load = load_data(False,batch_size,False)

    for epoch in range(epochs):
        total_loss = 0
        net.train()
        for X, y in train_load:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            total_loss += l
            l.backward()
            optimizer.step()
        test_acc = evaluate_accuracy(net,test_load)
        print('epoch:{},avg_loss:{:.3f},acc:{:.3f}'.format(epoch,total_loss/(batch_size),test_acc))


if __name__ == '__main__':

    train()