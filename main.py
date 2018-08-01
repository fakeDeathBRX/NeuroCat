#! /usr/bin/env python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#device = torch.device('cpu') # Uncomment to use only CPU

BATCH_SIZE = 10
EPOCH = 10
LR = 0.0001

train_data = torchvision.datasets.ImageFolder("./cnn/", transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()]))
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.ImageFolder("./test/", transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()]))
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=16,kernel_size=7,stride=1,padding=2),
                                    nn.ReLU(),nn.MaxPool2d(kernel_size=4, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),
                                    nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=1))
        self.out = nn.Linear(380192, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        out = self.out(x)
        return out, x

cnn = CNN().to(device)
print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss()
cnn.load_state_dict(torch.load('model.ckpt'))

steps_total = len(train_loader)
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        out = cnn(x)[0]
        loss = lossf(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if(step % 2 == 0):
        print('[Epoch #{}/{}]: Step: {}/{} | Loss:{:.4f}'
              .format(epoch+1,EPOCH,step,steps_total,loss.item()), end='\r')

torch.save(cnn.state_dict(), 'model.ckpt')
steps_total = len(test_loader)*BATCH_SIZE
correct = 0
for step, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    out = cnn(x)[0]
    prd = torch.max(out.data, 1)[1]
    correct += (prd == y).sum().item()
    print("[Test #{}/{}] Accuracy: {}%"
          .format(step,steps_total,(correct/(step+1))*100),end='\r')
print("[Test] Accuracy: {}/{} -> {}%".format(correct,steps_total,(correct/steps_total)*100))
#cnn()
