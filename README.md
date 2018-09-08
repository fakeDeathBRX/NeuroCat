# NeuroCat

#### A Convolutional Neural Network focused on distinguishing CAT(erpillar) from cats.

-------------

## Info

Have you ever mistaken a backhoe from a cat just because they're labeled alike? Luckly you will never have this problem again with this repo!

You can get about 90% accuracy with 10 epochs on default learning rate, this model is saved as `model.ckpt`.
Available Datasets:
1. Training (almost clean), containing about 600 images. [Download](https://drive.google.com/open?id=1yysLtJIyiBXXZj5XOAWsuI0ttljCrOrP).
2. Training extended (not clean), containing about 1900 images. [Download](https://drive.google.com/open?id=1npANkFgqCkGYn6gZdcuMyqdrEUqvyx-a).

> *All images were downloaded on Google Images using an extension.*

This project was made using [`PyTorch`](https://pytorch.org/).

## Usage

1. Clone this repo. `git clone https://github.com/fakeDeathBRX/NeuroCat`.

2. Download and extract the training dataset.

3. Run `python main.py`.

### Running options
| Argument | Description |
|--|--|
| train | Allow training based on `cnn` folder |
| test | Allow testing based on `test` folder |
| image image.ext | Get result of the CNN on `image.ext` |
| model name | Sets `name` as model |
| epoch N | Sets `N` epochs to run |
| lr N | Sets `LR = N` |
| batch N | Sets `BATCH_SIZE = N` |
| nload | Don't load the model |
| nsave | Don't save the model |
| gpu | Use GPU (default = cpu) |
| clear folder | Attempt to clear the `folder` dataset |
| class folder | Classify the images on `folder` |

### Usage examples

> python main.py class images

This will classify the `images` folder between CAT and cat.

> python main.py train epoch 10 gpu nload

This will train a new model `model.ckpt` with 10 epochs on GPU.

> python main.py train epoch 20 test class images model e20

This will train model `e20.ckpt` with 20 epochs, test and classify `images` folder.


## C.N.N. Specifications

You can check out the CNN specifications and edit them on `CNN.py`.

### Architecture:
| Layer | Description |
|--|--|
| Input | 224x224x3 |
| Conv1 | Dropout (20%) |
| | 16 kernels (7x7), stride = 1, padding = 2; |
| | ReLU;
| | MaxPool 4x4, stride = 2
| Conv2 | 32 kernels (5x5), stride = 1, padding = 2;
| | ReLU;
| | MaxPool 2x2, stride = 2
| FC | 366368 x 2 |

### Default Values:

`BATCH_SIZE = 10`,

`LEARNING_RATE = 0.0001`,

`EPOCH = 1`

-----------
