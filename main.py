#! /usr/bin/env python
import sys
import os
import glob
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
from torch.autograd import grad
from PIL import Image
from subprocess import call
from CNN import CNN

###     DEFAULTS
device = torch.device("cpu")
ModelName = "model.ckpt"
outfolder = "class/"

###     END

if(len(sys.argv) == 1 or "help" in sys.argv or "h" in sys.argv):
    print("Usage examples:")
    print("'python main.py train test epoch 5' to train on 5 epochs and test.")
    print("'python main.py nload train' to create a new model based on 1 training epoch.")
    print("'python main.py model newm train gpu' to load/save newm model on gpu based on 1 training epoch.")
    print("'python main.py test clear cnn1' to test and then clear the 'cnn1' folder dataset.")
    print("")
    print("Check README.md for more details.")
    sys.exit(0)

BATCH_SIZE = 10
EPOCH = 1
LR = 0.0001

LOAD = True
SAVE = True
TRAIN = False
TEST = False
CLEAR = False
CLASS = False
TestImage = ""
ClearFolder = ""
ClassFolder = ""

for i in range(len(sys.argv)):
    if(sys.argv[i] == "image"):
        TestImage = sys.argv[i+1]
        i += 1
    elif(sys.argv[i] == "epoch"):
        EPOCH = int(sys.argv[i+1])
        i += 1
    elif(sys.argv[i] == "batch"):
        BATCH_SIZE = int(sys.argv[i+1])
        i += 1
    elif(sys.argv[i] == "model"):
        ModelName = sys.argv[i+1] + ".ckpt"
        i += 1
    elif(sys.argv[i] == "clear"):
        ClearFolder = sys.argv[i+1]
        CLEAR = True
        if(not os.path.isdir(ClearFolder)):
            print("[ERROR] Clear folder '{}' does not exist!".format(ClearFolder))
            sys.exit(0)
        if(not os.path.isdir(ClearFolder + "/cat_0")):
            print("[WARNING] Clear data folder '{}/cat_0' does not exist!".format(ClearFolder))
        if(not os.path.isdir(ClearFolder + "/CAT_1")):
            print("[WARNING] Clear data folder '{}/CAT_1' does not exist!".format(ClearFolder))
        i += 1
    elif(sys.argv[i] == "class"):
        ClassFolder = sys.argv[i+1]
        CLASS = True
        if(not os.path.isdir(ClassFolder)):
            print("[ERROR] Classify folder '{}' does not exist!".format(ClassFolder))
            sys.exit(0)
        i += 1

if("train" in sys.argv): TRAIN = True
if("test" in sys.argv): TEST = True
if("nload" in sys.argv): LOAD = False
if("nsave" in sys.argv): SAVE = False
if("gpu" in sys.argv):
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        print("[Error]: No CUDA support for 'gpu' argument.")

def image_loader(image_name):
    t = transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
    image = Image.open(image_name)
    image = t(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device)

cnn = CNN().to(device)
#print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss()

if(LOAD):
    if(not os.path.isfile(ModelName)):
        print("[WARNING] Model '{}' does not exist. Creating one...".format(ModelName))
        torch.save(cnn.state_dict(), ModelName)
    cnn.load_state_dict(torch.load(ModelName))

if(TRAIN):
    for epoch in range(EPOCH):
        train_data = torchvision.datasets.ImageFolder("./cnn/", transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()]))
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        steps_total = len(train_loader)
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            out = cnn(x)
            loss = lossf(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[Epoch #{}/{}]: Step: {}/{} | Loss: {:.4f}"
                  .format(epoch+1,EPOCH,step,steps_total,loss.item()), end="\r")
    if(SAVE): # Save on each epoch
        torch.save(cnn.state_dict(), ModelName)
    print("[Train] Completed {} epochs.".format(EPOCH))

if(TEST):
    ccat0 = 0
    ccat1 = 0
    lcat0 = len(os.listdir("test/cat_0/"))
    lcat1 = len(os.listdir("test/CAT_1/"))
    for file in glob.iglob("test/cat_0/*"):
        it = image_loader(file)
        out = cnn(it)
        prd = torch.max(out.data, 1)[1]
        if(prd == 1):
            ccat0 += 1
        print("[Test] {}/{} right answers on cat.".format(ccat0,lcat0),end="\r")
    print("[Test] {}/{} right answers on cat.".format(ccat0,lcat0))
    for file in glob.iglob("test/CAT_1/*"):
        it = image_loader(file)
        out = cnn(it)
        prd = torch.max(out.data, 1)[1]
        if(prd == 0):
            ccat1 += 1
        print("[Test] {}/{} right answers on CAT.".format(ccat1,lcat1),end="\r")
    print("[Test] {}/{} right answers on CAT.".format(ccat1,lcat1))
    print("[Test] Total accuracy {}/{} -> {:.1f}%."
            .format(ccat0+ccat1,lcat0+lcat1,(ccat0+ccat1)/(lcat0+lcat1)*100))

if(TestImage != ""):
    it = image_loader(TestImage)
    out = cnn(it)
    prd = torch.max(out.data, 1)[1]
    if(prd == 0): print("[Image Test]: {} represents a CAT".format(TestImage))
    else: print("[Image Test]: {} represents a cat".format(TestImage))


"""     CLEAR
    This function will take a folder FLD and will clear the wrong labels based on a model.
    The folder FLD must be like this:
        - FLD/cat_0/*.ext
        - FLD/CAT_1/*.ext
    cat_0 = cat (animal), CAT_1 = machinery
"""
if(CLEAR):
    if(not os.path.isdir("wrong")):
        print("[WARNING] Clearing dump folder does not exit. Creating one...")
        os.makedirs("wrong")
        if(not os.path.isdir("wrong/cat0")):
            os.makedirs("wrong/cat0")
        if(not os.path.isdir("wrong/CAT1")):
            os.makedirs("wrong/CAT1")
    errado = 0
    if(os.path.isdir(ClearFolder + "/cat_0")):
        for file in glob.iglob(ClearFolder + "/cat_0/*"):
            it = image_loader(file)
            out = cnn(it)
            prd = torch.max(out.data, 1)[1]
            if(prd == 0):
                filename = file.split("/")[-1]
                errado += 1
                print("[Clear] Wrong images: {}".format(errado),end="\r")
                os.rename(file,"wrong/cat0/"+filename)
        print("[Clear] Cleared {} images on cat_0".format(errado))
        errado = 0
    if(os.path.isdir(ClearFolder + "/CAT_1")):
        for file in glob.iglob(ClearFolder + "/CAT_1/*"):
            it = image_loader(file)
            out = cnn(it)
            prd = torch.max(out.data, 1)[1]
            if(prd == 1):
                filename = file.split("/")[-1]
                errado += 1
                print("[Clear] Wrong images: {}".format(errado),end="\r")
                os.rename(file,"wrong/CAT1/"+filename)
        print("[Clear] Cleared {} images on CAT_1".format(errado))

if(CLASS):
    cat0 = 0
    CAT1 = 0
    if(not os.path.isdir(outfolder)):
        print("[WARNING] Classify output folder does not exit. Creating one...")
        os.makedirs(outfolder)
        if(not os.path.isdir(outfolder + "/cat0")):
            os.makedirs(outfolder + "/cat0")
        if(not os.path.isdir(outfolder + "/CAT1")):
            os.makedirs(outfolder + "/CAT1")
    for file in glob.iglob(ClassFolder + "/*"):
        it = image_loader(file)
        out = cnn(it)
        prd = torch.max(out.data, 1)[1]
        if(prd == 0): # CAT
            filename = file.split("/")[-1]
            os.rename(file,outfolder + "/CAT1/"+filename)
            CAT1 += 1
        if(prd == 1): # cat
            filename = file.split("/")[-1]
            os.rename(file,outfolder + "/cat0/"+filename)
            cat0 += 1
        print("[Classifier] Processed {} images. ({} CAT, {} cat)".format(cat0+CAT1,CAT1,cat0), end='\r')
    print("[Classifier] Processed {} images. ({} CAT, {} cat)".format(cat0+CAT1,CAT1,cat0))
