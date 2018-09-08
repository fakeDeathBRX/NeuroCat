import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( nn.Dropout(p=0.2),
                                    nn.Conv2d(in_channels=3,out_channels=16,kernel_size=7,stride=1,padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=4, stride=2))
        self.conv2 = nn.Sequential( nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=1))
        self.out1 =                 nn.Linear(366368, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        out = self.out1(x)
        return out
