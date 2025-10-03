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
 * @version: 2025-10-02
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef MODEL_CONFIG_HPP
#define MODEL_CONFIG_HPP

/* Standard dependencies */
#include <vector>
#include <string>
#include <string_view>
#include <ranges>
#include <iostream>
#include <cstdio>

/* Local dependencies */
#include "Log.hpp"

namespace Model_Config_NS {

/**
 * Enum to prevent repetitive conversion code between
 * strings and their numerical representations
 */
typedef enum {
    FLOAT,
    SIZE_T
} Flag_Conversion_Type;

/* Use macros here for easy expansion later */
#define FLOAT_SSCANF_FMT "%f"
#define SIZE_T_SSCANF_FMT "%zu"

typedef struct Model_Config {
    /**
     * Integer representation of the Cost_Function (can be cast)
     */
    int cost_function = 2;
    /**
     * Learning rate
     */
    float learning_rate = 0;
    /**
     * Lambda hyperparameter
     */
    float lambda = 0;
    /**
     * Number of layers and their corresponding number of neurons
     */
    std::vector<size_t> layer_info = std::vector<size_t>();
    /**
     * Number of images to train on from the dataset
     */
    size_t num_train = 0;
    /**
     * Numer of epochs to train on
     */
    size_t epochs = 0;
    /** 
     * Batch size 
     */
    size_t batch_size = 0;
    /**
     * Path to the training dataset (MNIST Images)
     */
    std::string training_dataset_path = std::string();
    /**
     * Path to the training labels (MNIST Labels)
     */
    std::string training_labels_path = std::string();
    /**
     * Path to write the model to after training (or read from)
     */
    std::string model_path = std::string();
} Model_Config;

/**
 * Parse a command line flag for training a model
 * @param flag String to parse
 * @param config Reference to a Model_Config to populate
 * @returns True if parsed properly, False if the config doesn't parse
 */
bool parse_flag(const std::string& flag, Model_Config& config);

/**
 * Convert a flag value to numerical representation
 * @param flag Const reference to a string to parse the value from
 * @param type Flag_Conversion_Type for the conversion itself
 * @param dest void* of the destination
 * @returns True if it's a valid conversion, False if it's invalid
 */
bool convert_flag_to_num(const std::string& flag, Flag_Conversion_Type type, void* dest);

/**
 * Validate all flags are set in the model config to start a training loop
 * @param config Reference to a Model_Config to check
 * @returns True if all required values are set, False if something is missing
 */
bool check_training_config(Model_Config& config);
};

#endif
