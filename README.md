# Intro

A project for generic fine-tuning/extending
(freeze CNN layers and connect with your own customzied FC layers)
pre-trained/existing models, such as AlexNet, VGG-16, ... (More CNN based).

# Manual

Step1. Put your datasets somewhere and train.csv under `/data/alexnet_finetune`

Step2. train.csv should be the format with first column to be the locations of images (I put an example in the folder)

Step3. Run:

```
# Download Pre-trained model
bash setup.sh

export PYTHONPATH='.'

# You can also take advantages of the Makefile, which actually inspires me dockerize this project if I have time
python apps/finetune_alexnet_train.py
```


# Keys

One dependencies we live on is the pre-trained [weights](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) from BVLC.
That said you don't wanna mess up with the name scope of those layers you wanna freeze. So be careful.
How we load the weights (`/services/weights_load_services`) should provide you enough information.

You are free to add as many layers as you like and just be aware that `conv1 ~ conv5` and `fc6~fc8` are those layers (as well as the NAME) you can load a pre-trained weights.

I try very hard to implement what I thought is the best practice for tensorflow -- separate:

- [X] architecture model (models/alexnet.py)

- [X] computation model (models/finetune_graph.py)

- [X] trainer model (models/train.py)

- [X] training app (apps/finetune_alexnet_train.py)

so that you can independently change any part of those without impacting other components.

A better example will be my anther repo -- [generic CNN in tensorflow](https://github.com/KleinYuan/cnn), which may have a better idea of what I am trying to do.
Since for this project, you need to sacrifice some graceful implementation due to the constraints of the pre-trained weights organization.


# references

1. [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
2. [Fine Tuning AlexNet on Tensorflow Example](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html#finetune)
3. [AlexNet Explainations in details](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
