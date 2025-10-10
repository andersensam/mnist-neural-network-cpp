# MNIST Neural Network in C++

This projects aims to create a basic, flexible neural network implementation in C++, that predicts the numbers depicted in handwritten images in the MNIST dataset.

This is a follow on to my original MNIST Neural Network in C, [available here](https://github.com/andersensam/mnist-neural-network-c). The original repo includes references, inspriations, and design philosophy.

### A Note on EMNIST

The original MNIST dataset is relatively small and contains only digits. As the [EMNIST (Extended MNIST)](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset is available in the same binary format as the Yann LeCun MNIST dataset (linked above), we can simply point to the EMNIST datasets' labels and images to train / run inference on them.

If using the digits dataset, there are no changes needed; **however**, to add in letters, `include/MNIST_Utils.hpp` needs to be modified. The line here must be adjusted:

```
#define MNIST_LABELS 10
```

If using the EMIST balanced dataset, change to the following:

```
#define MNIST_LABELS 47
```

No other changes are required beyond the above.

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
    --model_path=[path to write model to]
```

A complete example is listed below:
```
./mnist-neural-network online-training --layers=784,100,10 --learning_rate=0.1 \
    --lambda=0.1 --cost_function=quadratic --num_train=3000 --epochs=1 \
    --dataset_path=../data/train-images-idx3-ubyte --labels_path=../data/train-labels-idx1-ubyte \
    --model_path=../models/test.model
```

The example reveals that: we want to train a new model, using a **learning rate** of 0.1, there are **3** total layers (including input). The first layer has 784 neurons, the second has 100, and the output has 10. We want to train on a subset of 3000 images and we want to export the model to file `test.model`.

Running the example above produces the following output:

```
$ ./mnist-neural-network online-training --layers=784,100,10 --learning_rate=0.1 \
    --lambda=0.1 --cost_function=quadratic --num_train=3000 --epochs=1 \
    --dataset_path=../data/train-images-idx3-ubyte --labels_path=../data/train-labels-idx1-ubyte \
    --model_path=../models/test.model

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