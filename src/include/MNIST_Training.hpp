/*  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Basic Neural Network in C++
 * @author : Samuel Andersen
 * @version: 2025-07-22
 *
 * General Notes:
 *
 * TODO: 
 */

#ifndef MNIST_TRAINING_HPP
#define MNIST_TRAINING_HPP

#define MNIST_TRAINING_SHOW_LOSS 1
#define MNIST_TRAINING_SHOW_LOSS_STEPS 100
#define MNIST_TRAINING_SHOW_BATCH_LOSS_STEPS 100

/* Standard dependencies */

/* Local dependencies */
#include "Log.hpp"
#include "Matrix.hpp"
#include "MNIST_Utils.hpp"
#include "Neural_Network.hpp"

namespace MNIST_Training_NS {

/**
 * Train a new model using online training (batch size of 1), saving it to a file when it completes
 * @param labels_path Path to the labels file to read
 * @param images_path Path to the images file
 * @param layer_info A reference to std::vector<size_t> containing the number of neurons in each layer
 * @param learning_rate Learning rate hyperparameter
 * @param lambda Normalization hyperparameter
 * @param num_training_images Number of images from the dataset to train on
 * @param epochs Number of epochs to run across the entire dataset
 * @param cost_function The cost function to use (quadratic or cross-entropy)
 * @param model_path Path to save the model once it has been run
 */
void train_new_model(const char* labels_path, const char* images_path, 
    const std::vector<size_t>& layer_info, float learning_rate, float lambda, size_t num_training_images, 
    size_t epochs, Neural_Network_NS::Cost_Function cost_function, const char* model_path);

/**
 * Shuffle the indicies used for pulling images and labels
 * See Fisher-Yates Shuffle: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
 * @param index Array of size_t to shuffle
 * @param elements Number of elements in the index array
 */
void shuffle(size_t* index, size_t elements);

/**
 * Generate an array of size_t that are randomly shuffled
 * @param elements Number of elements
 * @returns Returns an array of size_t
 */
size_t* create_index_array(size_t elements);


};

#endif
