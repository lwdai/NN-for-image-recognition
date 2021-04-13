# NN-for-image-recognition
> In this project, I have built and trained neural networks to solve the image classification problem on the CIFAR10 dataset. 
First I reimplemented a model provided by Tensorflow with two convolutional layers as a baseline. Next I reshaped the network by stacking multiple convolutional layers. 
I also attempted to make the network deeper by adding network-innetwork layers and using fractional max pooling. My best result
beat the baseline with 86% accuracy on the test set as well as a
much faster speed of convergence.

> Index Terms—image classification, convolution, neural networks, deep learning, artificial intelligence

> [Complete Report](report/Liwen-project.pdf)

====================
> Need to have GPU version of tensorflow installed: https://www.tensorflow.org/  
> Tested on python 2.7   

> Put uncompressed CIFAR-10 binary files under ./cifar10  https://www.cs.toronto.edu/~kriz/cifar.html
> Try the two scripts to run

> For help
> $python CNN/cnn_main.py --help

> Performance: use --model='CNN_deep'. 86% accuracy in 2 hours on GTX 770M.  
> Thanks to https://github.com/ry/tensorflow-resnet, I had a starting point for my code.
