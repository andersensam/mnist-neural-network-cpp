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

#include "include/MNIST_Training.hpp"

using Matrix = Matrix_NS::Matrix<float>;
using Neural_Network = Neural_Network_NS::Neural_Network;
using MNIST_Images = MNIST_Utils_NS::MNIST_Images;
using MNIST_Labels = MNIST_Utils_NS::MNIST_Labels;

void MNIST_Training_NS::train_new_model(const char* labels_path, const char* images_path, 
    const std::vector<size_t>& layer_info, float learning_rate, float lambda, size_t num_training_images, 
    size_t epochs, Neural_Network_NS::Cost_Function cost_function, const char* model_path) {

    MNIST_Images images = MNIST_Images(images_path);
    MNIST_Labels labels = MNIST_Labels(labels_path);

    if (layer_info.size() == 0) {
        Log::log_message(Log::Log_Priority::ERROR, "train_new_model",
            "Invalid layer_info vector provided");
        return;
    }

    // Instantiate the Neural Network
    Neural_Network nn = Neural_Network(layer_info, learning_rate, lambda, cost_function);

    // Setup Matrix instances that will be reused for processing images and labels
    Matrix current_image = Matrix(MNIST_IMAGE_SIZE, 1);
    Matrix current_label = Matrix(MNIST_LABELS, 1);

    // Setup a shuffled array index
    size_t* shuffled_index = NULL;
    // Track the loss
    float loss = 0;

    for (size_t i = 0; i < epochs; ++i) {

        // Create a shuffled index array
        shuffled_index = create_index_array(images.size());

        // Iterate through the number of images per epoch
        for (size_t j = 0; j < num_training_images; ++j) {
            
            images.get_flat(shuffled_index[i], current_image);
            labels.create_label(shuffled_index[i], current_label);
            loss = nn.train(current_image, current_label, num_training_images);

            if (MNIST_TRAINING_SHOW_LOSS) {
                if (j % MNIST_TRAINING_SHOW_LOSS_STEPS == 0) {
                    Log::log_message(Log::Log_Priority::INFO, "train_new_model",
                        std::format("Online trainer step {} loss={}", j, loss));
                }
            }
        }
        free(shuffled_index);
    }
    nn.save(model_path);
}

void MNIST_Training_NS::shuffle(size_t* index, size_t elements) {

    if (index == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "shuffle",
            "Invalid index array provided");
        exit(EXIT_FAILURE);
    }

    size_t random_index = 0;
    size_t current_value = 0;

    for (size_t i = elements - 1; i > 0; --i) {
        random_index = rand() % (i + 1);
        // Grab the current value and swap with the value @ random_index
        current_value = index[i];
        index[i] = index[random_index];
        index[random_index] = current_value;
    }
}

size_t* MNIST_Training_NS::create_index_array(size_t elements) {

    size_t* target = (size_t*)calloc(elements, sizeof(size_t));

    if (target == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "create_index_array",
            "Unable to allocate memory for the index array");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < elements; ++i) {
        target[i] = i;
    }
    // Shuffle the index array
    shuffle(target, elements);
    return target;
}
