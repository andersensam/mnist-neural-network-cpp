# MNIST Neural Network in C++

This projects aims to create a basic, flexible neural network implementation in C, that predicts the numbers depicted in handwritten images in the MNIST dataset.

There are a number of inspirations for this project, including:

AndrewCarterUK's [repo](https://github.com/AndrewCarterUK/mnist-neural-network-plain-c)

3 Blue 1 Brown's [YouTube series on neural networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

Mark Kraay's [YouTube video](https://www.youtube.com/watch?v=ReOxVMxS83o) and [accompanying repo](https://github.com/markkraay/mnist-from-scratch)

CyberZHG's [repo on layer normalization](https://github.com/CyberZHG/torch-layer-normalization)

## Approach to this Neural Network

This project is a naive attempt at building a simple neural network and has a few 'shortcuts' taken, including a lack of multithreaded training and GPU utilization.

I hope to update this repo as time goes on with adjustments to the error, activation, and training methods.

I have tried to make the underlying code as flexible as possible, allowing for quick adjustments to the number of layers, neurons, biases, etc. To see my convoluted implementation of a matrix library in C, check out [Matrix](https://github.com/andersensam/Matrix).

As you may expect, the models exported from this neural network cannot be used with any other existing framework that I'm aware of. I had fun coming up with a different file format based on the [Yann LeCun MNIST dataset](https://yann.lecun.com/exdb/mnist/).

### A Note on EMNIST

The original MNIST dataset is relatively small and contains only digits. As the [EMNIST (Extended MNIST)](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset is available in the same binary format as the Yann LeCun MNIST dataset (linked above), we can simply point to the EMNIST datasets' labels and images to train / run inference on them.

If using the digits dataset, there are no changes needed; **however**, to add in letters, `include/MNIST_Labels.h` needs to be modified. The line here must be adjusted:

```
#define MNIST_LABELS 10
```

If using the EMIST balanced dataset, change to the following:

```
#define MNIST_LABELS 47
```

No other changes are required beyond the above.

## Compilation

To build the neural network, simply use `make`. By default, this will compile with `-O3`

To disable optimizations and compile with debug symbols, use `make debug`

## Using the Neural Network

The main binary has three options available (with suboptions for train and predict):

1. Train: train the model on a specified set of images and labels

2. Predict: run inference on a set of images and validate against labels

3. Help: show the help menu

### Training the Neural Network (online training)

As noted above, the first option is to train a new neural network from scratch. To get started, examine the following syntax:

```
./target/main train <path to labels> <path to images> <learning rate> <use biases> <number of layers> <[neurons in each layer]> <images to train on> <epochs> <model name>
```

A complete example is listed below:
```
./target/main train data/train-labels-idx1-ubyte data/train-images-idx3-ubyte 0.1 true 3 784 100 10 1500 2 small_100.model
```

The example reveals that: we want to train a new model, using a **learning rate** of 0.1, we want to use biases, there are **3** total layers (including input). The first layer has 784 neurons, the second has 100, and the output has 10. We want to train on a subset of 1500 images and we want to export the model to file `small_100.model`.

Running the example above produces the following output:

```
$ ./target/main train data/train-labels-idx1-ubyte data/train-images-idx3-ubyte 0.1 true 3 784 100 10 1500 2 small_100.model

[2024-09-18 23:32:59]: Starting to load MNIST labels
[2024-09-18 23:32:59]: Finished loading MNIST labels
[2024-09-18 23:32:59]: Starting to load MNIST images
[2024-09-18 23:33:00]: Finished loading MNIST images
[2024-09-18 23:33:00]: Starting model training
[2024-09-18 23:33:00]: Finished model training
[2024-09-18 23:33:00]: Saving model
[2024-09-18 23:33:00]: Finished saving model
```

### (Mini)batch Training of the Neural Network

Using (mini)batch, we can train the Neural Network with the following syntax:

```
./target/main train <path to labels> <path to images> <learning rate> <use biases> <number of layers> <[neurons in each layer]> <images to train on> <batch size> <epochs> <model name>
```

A complete example is listed below:

```
./target/main batch-train data/train-labels-idx1-ubyte data/train-images-idx3-ubyte 0.1 false 3 784 100 10 10000 10 3 small_100_batch.model
```

In the example above, we use a **learning rate** of 0.1, we do not use any biases (all initialized to 0), we train on 10000 images, with a batch size of 10, and we run through the full dataset 3 times. We save the result to `small_100_batch.model`.

Running the above results in:

```
$ ./target/main batch-train data/train-labels-idx1-ubyte data/train-images-idx3-ubyte 0.1 false 3 784 100 10 10000 10 3 small_100_batch.model

[2024-09-25 21:03:38]: Starting to load MNIST labels
[2024-09-25 21:03:38]: Finished loading MNIST labels
[2024-09-25 21:03:38]: Starting to load MNIST images
[2024-09-25 21:03:39]: Finished loading MNIST images
[2024-09-25 21:03:39]: Starting model training
[2024-09-25 21:03:43]: Finished model training
[2024-09-25 21:03:43]: Saving model
[2024-09-25 21:03:43]: Finished saving model
```

As you can set the batch size to whatever you'd like, you could use the batch size as the total training image size for full batch training.

### Running Inference

Running inference on a trained model follows a similar syntax:

```
./target/main predict <path to labels> <path to images> <images to predict> <path to model>
```

An example is:

```
./target/main predict data/t10k-labels-idx1-ubyte data/t10k-images-idx3-ubyte 1000 models/large_100.model
```

Executing the above results in:

```
$ /target/main predict data/t10k-labels-idx1-ubyte data/t10k-images-idx3-ubyte 1000 models/large_100.model

[2024-09-18 23:39:04]: Starting to load MNIST labels
[2024-09-18 23:39:04]: Finished loading MNIST labels
[2024-09-18 23:39:04]: Starting to load MNIST images
[2024-09-18 23:39:05]: Finished loading MNIST images
[2024-09-18 23:39:05]: Starting to load model from file
[2024-09-18 23:39:05]: Finished loading model from file
[2024-09-18 23:39:05]: Starting inference
[2024-09-18 23:39:05]: Finished inference

Statistics:
Model path: models/large_100.model
Images predicted: 1000
Images predicted correctly: 946
Percentage correct: 94.60000%
```

### Multithreaded Inference

Running multithreaded inference is the same as standard inference, only swapping out `predict` for `threaded-predict`:

```
./target/main threaded-predict <path to labels> <path to images> <images to predict> <path to model>
```

An example is:

```
./target/main threaded-predict data/t10k-labels-idx1-ubyte data/t10k-images-idx3-ubyte 1000 models/large_256_32.model
```

Executing the above results in:
```
$ ./target/main threaded-predict data/t10k-labels-idx1-ubyte data/t10k-images-idx3-ubyte 1000 models/large_256_32.model

[2024-09-22 14:25:24]: Starting to load MNIST labels
[2024-09-22 14:25:24]: Finished loading MNIST labels
[2024-09-22 14:25:24]: Starting to load MNIST images
[2024-09-22 14:25:25]: Finished loading MNIST images
[2024-09-22 14:25:25]: Starting to load model from file
[2024-09-22 14:25:25]: Finished loading model from file
[2024-09-22 14:25:25]: Starting threaded inference
[2024-09-22 14:25:33]: Finished threaded inference

Statistics:
Model path: models/large_256_32.model
Images predicted: 10000
Images predicted correctly: 9228
Percentage correct: 92.28000%
```

**Note:** To adjust the number of threads used in `threaded-predict`, please adjust `INFERENCE_MAX_THREADS` in `main.h`. Its default value is 4.

## File Descriptions

Descriptions of each file in `src/` and their functions:

### *main.c*

The main binary, used for training or predicting.

### *Matrix.c*

The Matrix library. See note above to check out the upstream repo.

### *MNIST_Images.c*

A data type created to store the MNIST images in `float` format, wrapping inside a `floatMatrix`. It also handles importing the dataset.

### *MNIST_Labels.c*

Read in the label dataset and wrap it in a container.

### *Neural_Network.c*

The Neural Network itself and associated helper functions / data structures, like the Neural Network Layer structure.

### *utils.c*

Small, useful utilities