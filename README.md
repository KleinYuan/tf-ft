# Intro

A project for generic fine-tuning/extending
(freeze CNN layers and connect with your own customzied FC layers)
pre-trained/existing models, such as AlexNet, VGG-16, ... (More CNN based).

# Manual

Step1. Put your datasets somewhere and train.csv under `/data/alexnet_finetune`

Step2. train.csv should be the format with first column to be the locations of images

Step3. Run:

```
# Download Pre-trained model
bash setup.sh

export PYTHONPATH='.'
python apps/finetune_alexnet_train.py
```

# references

1. [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
2. [Fine Tuning AlexNet on Tensorflow Example](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html#finetune)
3. [AlexNet Explainations in details](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
