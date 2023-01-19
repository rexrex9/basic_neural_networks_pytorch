
from torch.utils.data import DataLoader
import torchvision
from data_set import filepaths as fp

train_data = torchvision.datasets.MNIST(
    root=fp.MNIST,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.MNIST(
    root=fp.MNIST,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

#print(train_data)
#print(test_data)
#

train_load =  DataLoader(dataset=train_data,batch_size=16,shuffle=True)
train_x,train_y = next(iter(train_load))
#print(train_x[0].shape)
#print(train_x[0])

# pip install python-opencv
import cv2
img = torchvision.utils.make_grid(train_x,nrow=10)
img = img.numpy().transpose(1,2,0)
cv2.imshow('img',img)
cv2.waitKey()