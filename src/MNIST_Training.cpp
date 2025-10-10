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
 * @version: 2025-10-09
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

void MNIST_Training_NS::train_new_model_online(const Model_Config_NS::Model_Config& config) {

    MNIST_Images images = MNIST_Images(config.dataset_path);
    MNIST_Labels labels = MNIST_Labels(config.labels_path);

    if (config.layer_info.size() == 0) {
        Log::log_message(Log::Log_Priority::ERROR, "train_new_model",
            "Invalid layer_info vector provided");
        return;
    }

    // Instantiate the Neural Network
    Neural_Network nn = Neural_Network(config.layer_info, config.learning_rate, 
        config.lambda, static_cast<Neural_Network_NS::Cost_Function>(config.cost_function));

    // Setup Matrix instances that will be reused for processing images and labels
    Matrix current_image = Matrix(MNIST_IMAGE_SIZE, 1);
    Matrix current_label = Matrix(MNIST_LABELS, 1);

    // Setup a shuffled array index
    size_t* shuffled_index = NULL;
    // Track the loss
    float loss = 0;

    // Track the total steps across epochs
    size_t current_step = 0;

    for (size_t i = 0; i < config.epochs; ++i) {

        // Create a shuffled index array
        shuffled_index = create_index_array(images.size());

        // Iterate through the number of images per epoch
        for (size_t j = 0; j < config.num_train; ++j) {
            
            images.get_flat(shuffled_index[i], current_image);
            labels.create_label(shuffled_index[i], current_label);
            loss = nn.train(current_image, current_label, config.num_train);

            if (MNIST_TRAINING_SHOW_LOSS) {
                if (j % MNIST_TRAINING_SHOW_LOSS_STEPS == 0) {
                    Log::log_message(Log::Log_Priority::INFO, "train_new_model_online",
                        std::format("Online trainer epoch {} step {} loss={}", i, current_step, loss));
                }
            }

            ++current_step;
        }
        free(shuffled_index);
    }
    nn.save(config.model_path.c_str());
}

void MNIST_Training_NS::train_new_model_batch(const Model_Config_NS::Model_Config& config) {

    MNIST_Images images = MNIST_Images(config.dataset_path);
    MNIST_Labels labels = MNIST_Labels(config.labels_path);

    if (config.layer_info.size() == 0) {
        Log::log_message(Log::Log_Priority::ERROR, "train_new_model",
            "Invalid layer_info vector provided");
        return;
    }

    // Instantiate the Neural Network
    Neural_Network nn = Neural_Network(config.layer_info, config.learning_rate, 
        config.lambda, static_cast<Neural_Network_NS::Cost_Function>(config.cost_function));

    // Setup Matrix instances that will be reused for processing images and labels
    Matrix current_images = Matrix(MNIST_IMAGE_SIZE, config.batch_size);
    Matrix current_labels = Matrix(MNIST_LABELS, config.batch_size);

    // Store information about the current batch and step number
    size_t steps_per_epoch = config.num_train / config.batch_size;
    size_t final_batch_size = config.num_train % config.batch_size;

    // Setup a shuffled array index
    size_t* shuffled_index = NULL;
    // Track the loss
    float loss = 0;

    // Track the total number of steps across epochs
    size_t current_step = 0;

    for (size_t i = 0; i < config.epochs; ++i) {

        // Create a shuffled index array
        shuffled_index = create_index_array(images.size());

        // Iterate through the number of images per epoch
        for (size_t j = 0; j < steps_per_epoch; ++j) {

            if ((j == steps_per_epoch - 1) && (final_batch_size != 0)) {

                // Ensure we capture whatever the oddly sized final batch has
                Matrix final_images = Matrix(MNIST_IMAGE_SIZE, final_batch_size);
                Matrix final_labels = Matrix(MNIST_LABELS, final_batch_size);

                images.create_images_from_range(config.num_train - final_batch_size - 1, config.num_train - 1, final_images);
                labels.create_labels_from_range(config.num_train - final_batch_size - 1, config.num_train - 1, final_labels);
                loss = nn.batch_train(final_images, final_labels, config.num_train);
            }
            else {
            
                images.create_images_from_range(j * config.batch_size, ((j + 1) * config.batch_size) - 1, current_images);
                labels.create_labels_from_range(j * config.batch_size, ((j + 1) * config.batch_size) - 1, current_labels);
                loss = nn.batch_train(current_images, current_labels, config.num_train);
            }

            if (MNIST_TRAINING_SHOW_LOSS) {
                if (j % MNIST_TRAINING_SHOW_LOSS_STEPS == 0) {
                    Log::log_message(Log::Log_Priority::INFO, "train_new_model_batch",
                        std::format("Batch trainer epoch {} step {} loss={}", i, current_step, loss));
                }
            }

            ++current_step;
        }
        free(shuffled_index);
    }
    nn.save(config.model_path.c_str());
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

    size_t* target = static_cast<size_t*>(calloc(elements, sizeof(size_t)));

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
