
# NeuroCat

#### A Convolutional Neural Network mainly focused on distinguishing CAT(erpillar) from cats.

-------------

## Info

Have you ever mistaken a backhoe from a cat just because they're labeled alike? Luckly you will never have this problem again with this repo!

10 Epochs with default `LR` resulted on 96% (48/50) accuracy on the test set (25 images each). This model is saved on the repo (`model.ckpt`).

You can download the training dataset [here](https://drive.google.com/file/d/1YsbOcrYKytlqMM7CZolV-Evp8JGk56pE/view?usp=sharing), it contains 1420 images of cats and 1425 images of Caterpillar machinery, totalizing 2845 images on training.

On the test dataset, there are 25 images of each type, totalizing 50 images.

  

> All those images were downloaded on Google Images using an extension.

  

This project is using [`PyTorch`](https://pytorch.org/).

## Using

1. Download this repo and the training dataset.

2. Extract the `dataset.zip` into the folder.

3. Configure `LOAD`,`SAVE`,`TRAIN` variables on `main.py` (all true by default).

4. Run `python main.py`.

  

## C.N.N. Specifications

### Architecture:
| Layer | Description |
|--|--|
| Input | 224x224x3 |
| Conv1 | 16 kernels (7x7), stride = 1, padding = 2; |
| | ReLU;
| | MaxPool 4x4, stride = 2
| Conv2 | 32 kernels (5x5), stride = 1, padding = 2;
| | ReLU;
| | MaxPool 2x2, stride = 2
| FC | 380192 x 2  

### Values:

`BATCH_SIZE = 10`,

*Note: this variable needs to have a number divisible by 50 (amount of test images), so the final results don't end up wrong.*

`LEARNING_RATE = 0.0001`,

As the training dataset is plenty, you can have numbers lower than ``0.001``

`EPOCH = 10`

  

-----------

  

> Note: ANN works faster with GPU support, if yours doesn't support newer CUDA versions, you may have to compile PyTorch