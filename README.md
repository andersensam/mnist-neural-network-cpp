# MNIST Neural Network in C++

This projects aims to create a basic, flexible neural network implementation in C++, that predicts the numbers depicted in handwritten images in the MNIST dataset.

This is a follow on to my original MNIST Neural Network in C, [available here](https://github.com/andersensam/mnist-neural-network-c). The original repo includes references, inspriations, and design philosophy.

### A Note on EMNIST

The original MNIST dataset is relatively small and contains only digits. As the [EMNIST (Extended MNIST)](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset is available in the same binary format as the Yann LeCun MNIST dataset (linked above), we can simply point to the EMNIST datasets' labels and images to train / run inference on them.

Using EMNIST requires the following option:
```
--module=emnist
```

## Compilation

To build the neural network, simply use `cmake`. By default, this will compile with `-O3`

Generally:
```
mkdir build
cd build && cmake ..
./mnist-neural-network [options here]
```

## Using the Neural Network

The main binary has three options available (with suboptions for train and predict):

1. Train: train the model on a specified set of images and labels

2. Predict: run inference on a set of images and validate against labels

3. Help: show the help menu

### Training the Neural Network (online training)

As noted above, the first option is to train a new neural network from scratch. To get started, examine the following syntax:

```
./mnist-neural-network online-training --layers=[comma separated layers] --learning_rate=[float value] \
    --lambda=[float value] --cost_function=[quadratic | cross_entropy] --num_train=[int value] --epochs=[int value] \
    --dataset_path=[path to training dataset] --labels_path=[path to training labels] \
    --model_path=[path to write model to] --module=[mnist | emnist]
```

A complete example is listed below:
```
./mnist-neural-network online-training --layers=784,100,10 --learning_rate=0.1 \
    --lambda=0.1 --cost_function=quadratic --num_train=3000 --epochs=1 \
    --dataset_path=../data/train-images-idx3-ubyte --labels_path=../data/train-labels-idx1-ubyte \
    --model_path=../models/test.model --module=mnist
```

The example reveals that: we want to train a new model, using a **learning rate** of 0.1, there are **3** total layers (including input). The first layer has 784 neurons, the second has 100, and the output has 10. We want to train on a subset of 3000 images and we want to export the model to file `test.model`.

Running the example above produces the following output:

```
$ ./mnist-neural-network online-training --layers=784,100,10 --learning_rate=0.1 \
    --lambda=0.1 --cost_function=quadratic --num_train=3000 --epochs=1 \
    --dataset_path=../data/train-images-idx3-ubyte --labels_path=../data/train-labels-idx1-ubyte \
    --model_path=../models/test.model --module=mnist

2025-10-10 10:55:38: [INFO] - <main::main>: Starting up... options provided:
online-training
--layers=784,100,10
--learning_rate=0.1
--lambda=0.1
--cost_function=quadratic
--num_train=3000
--epochs=1
--dataset_path=../data/train-images-idx3-ubyte
--labels_path=../data/train-labels-idx1-ubyte
--model_path=../models/test.model
--module=mnist
2025-10-10 10:55:38: [INFO] - <main::main>: Online training mode selected. Checking trainer config...
2025-10-10 10:55:38: [INFO] - <main::main>: Trainer config validated. Starting training loop...
2025-10-10 10:55:38: [INFO] - <MNIST_Images::MNIST_Images>: Reading 60000 images
2025-10-10 10:55:40: [INFO] - <MNIST_Labels::MNIST_Labels>: Reading 60000 labels
2025-10-10 10:55:40: [INFO] - <train_new_model_online>: Online trainer epoch 0 step 0 loss=2.880098
2025-10-10 10:55:41: [INFO] - <train_new_model_online>: Online trainer epoch 0 step 100 loss=0.004494261
2025-10-10 10:55:41: [INFO] - <train_new_model_online>: Online trainer epoch 0 step 200 loss=0.0021419716
```

### (Mini)batch Training of the Neural Network

Using (mini)batch, we can train the Neural Network with the following syntax:

```
./mnist-neural-network online-training --layers=[comma separated layers] --learning_rate=[float value] \
    --lambda=[float value] --cost_function=[quadratic | cross_entropy] --num_train=[int value] --epochs=[int value] \
    --batch_size[int value] \
    --dataset_path=[path to training dataset] --labels_path=[path to training labels] \
    --model_path=[path to write model to] --module=[mnist | emnist]
```

A complete example is listed below:

```
./mnist-neural-network batch-training --layers=784,100,10 --learning_rate=0.1 \
    --lambda=0.1 \--cost_function=cross_entropy --num_train=30000 --epochs=2 \
    --batch_size=4 \
    --dataset_path=../data/train-images-idx3-ubyte --labels_path=../data/train-labels-idx1-ubyte \
    --model_path=../models/test.model --module=mnist

```

In the example above, we use a **learning rate** of 0.1, we do not use any biases (all initialized to 0), we train on 30000 images, with a batch size of 4, and we run through the full dataset 2 times. We save the result to `test.model`.

Running the above results in:

```
$ ./mnist-neural-network batch-training --layers=784,100,10 --learning_rate=0.1 --lambda=0.1 \
    --cost_function=cross_entropy --num_train=30000 --epochs=2 \
    --dataset_path=../data/train-images-idx3-ubyte --labels_path=../data/train-labels-idx1-ubyte \
    --model_path=../models/test.model --batch_size=4 --module=mnist

2025-11-03 19:49:02: [INFO] - <main::main>: Starting up... options provided:
batch-training
--layers=784,100,10
--learning_rate=0.1
--lambda=0.1
--cost_function=cross_entropy
--num_train=30000
--epochs=2
--dataset_path=../data/train-images-idx3-ubyte
--labels_path=../data/train-labels-idx1-ubyte
--model_path=../models/test.model
--batch_size=4
--module=mnist
2025-11-03 19:49:02: [INFO] - <main::main>: Batch training mode selected. Checking trainer config...
2025-11-03 19:49:02: [INFO] - <main::main>: Trainer config validated. Starting training loop...
2025-11-03 19:49:02: [INFO] - <MNIST_Images::MNIST_Images>: Reading 60000 images
2025-11-03 19:49:03: [INFO] - <MNIST_Labels::MNIST_Labels>: Reading 60000 labels
2025-11-03 19:49:03: [INFO] - <MNIST_Training::train_new_model_batch>: Batch trainer epoch 0 step 0 loss=0.2870974
2025-11-03 19:49:03: [INFO] - <MNIST_Training::train_new_model_batch>: Batch trainer epoch 0 step 100 loss=0.37227955
```

As you can set the batch size to whatever you'd like, you could use the batch size as the total training image size for full batch training.

### Running Inference

Running inference on a trained model follows a similar syntax:

```
./mnist-neural-network inference --num_inference=[int value] --model_path=[path to existing model] \
    --dataset_path=[path to dataset] --labels_path=[path to labels] --module=[mnist | emnist]
```

An example is:

```
./mnist-neural-network inference --num_inference=10000 --model_path=../models/emnist.model \
    --dataset_path=../data/emnist-balanced-test-images-idx3-ubyte \
    --labels_path=../data/emnist-balanced-test-labels-idx1-ubyte --module=emnist
```

Executing the above results in:

```
$ ./mnist-neural-network inference --num_inference=10000 --model_path=../models/emnist.model \
    --dataset_path=../data/emnist-balanced-test-images-idx3-ubyte \
    --labels_path=../data/emnist-balanced-test-labels-idx1-ubyte --module=emnist

2025-11-10 16:58:53: [INFO] - <main::main>: Starting up... options provided:
inference
--num_inference=10000
--model_path=../models/emnist.model
--dataset_path=../data/emnist-balanced-test-images-idx3-ubyte
--labels_path=../data/emnist-balanced-test-labels-idx1-ubyte
--module=emnist
2025-11-10 16:58:53: [INFO] - <main::main>: Inference mode selected. Loading model...
2025-11-10 16:58:53: [INFO] - <main::main>: Inference config validated. Starting inference loop...
2025-11-10 16:58:53: [INFO] - <MNIST_Images::MNIST_Images>: Reading 18800 images
2025-11-10 16:58:54: [INFO] - <MNIST_Labels::load_labels>: Reading 18800 labels
2025-11-10 16:58:56: [INFO] - <MNIST_Inference::inference>: Inference results: [7317 / 10000] correct. 73.17% accuracy rate.
```

### Multithreaded Inference

Multithreaded inference in not currently implemented for the C++ version of this project.

## File Descriptions

Descriptions of each file in `src/` and their functions:

### *main.cpp*

The main binary, used for training or predicting.

### *Log.cpp*

A simple logging implementation.

### *MNIST_Inference.cpp*

Functions and utilities for running inference on an existing model.

### *MNIST_Training.cpp*

Training implementation for MNIST.

### *MNIST_Utils.cpp*

Useful tools for the MNIST dataset, such as loading the images and labels.

### *Model_Config.cpp*

Configuration checker and wrapper for running training or inference.

### *Neural_Network_Layer.cpp*

Implementation for each layer of a Neural Network.

### *Neural_Network.cpp*

The main implementation of the Neural Network.
