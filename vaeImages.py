import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import time

start_time = time.time()

IMG_W = 90
IMG_H = 90
LR = 1e-3
num_epochs = 600
BATCH_SIZE = 512

# 训练VAE模型图片路径
train_dir=r'F:\AAE\Images10000'
# 保存用于训练VAE的所有图片路径
train_images = []
for i in os.listdir(train_dir):
    train_images.append(os.path.join(train_dir,i))

# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    # 通过Normalize计算过后，将数据归一化到[-1,1]
    transforms.Normalize(mean=0.5, std=0.5)
])

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, IMG_W, IMG_H)
    return x

# 自定义训练集
class MyTrainset(Dataset):
    imgs = []
    def __init__(self, transform=img_transform, target_transform=None):
        self.imgs = train_images
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        image=self.imgs[index]
        img=Image.open(image).convert('1')
        img=img.resize((IMG_W, IMG_H))
        if self.transform is not None:
            img=self.transform(img)
        return img
trainSet = MyTrainset()    # 实例化自定义数据集
dataloader = DataLoader(dataset=trainSet, batch_size=BATCH_SIZE, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(8100, 4000)
        self.fc21 = nn.Linear(4000, 16)
        self.fc22 = nn.Linear(4000, 16)
        self.fc3 = nn.Linear(16, 4000)
        self.fc4 = nn.Linear(4000, 8100)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self,z):
        dh1=F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(dh1))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
model = VAE()

if torch.cuda.is_available():
    model.cuda()
reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)    # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        all_batchs=int(len(dataloader.dataset)/BATCH_SIZE)
        img = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(img),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                loss.item() / len(img)))
        if batch_idx == all_batchs:
            if not os.path.exists("./originalImages"):
                os.mkdir("./originalImages")
            if not os.path.exists("./reconstructed"):
                os.mkdir("reconstructed")
            save_image(to_img(img[-10:].data), f'originalImages/ori_image_{epoch}_{batch_idx}.png')
            save = to_img(recon_batch[-10:].data)
            save_image(save,
                    f'reconstructed/recon_image_{epoch}_{batch_idx}.png')
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))

torch.save(model.state_dict(), './vae.pth')
end_time = time.time()
print("耗时:%d" % ((end_time - start_time)/60) + "分钟")