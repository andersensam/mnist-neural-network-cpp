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
 * TODO: Continue adding functionality 
 */

#include "include/Model_Config.hpp"
using Model_Config_NS::Flag_Conversion_Type;

bool Model_Config_NS::parse_flag(const std::string& flag, Model_Config& config) {

    // Check for the presence of '=' in the flag value
    size_t flag_pos = flag.find("=", 0);

    // If we don't find '=', we know we have an invalid flag, regardless of what preceeds
    // the '=' sign
    if (flag_pos == std::string::npos) {
        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::parse_flag",
            "Unable to parse flag");
        return false;
    }

    // Split the flag value from the flag itself to prevent writing this piece
    // out for each flag option
    const std::string& flag_value = flag.substr(flag_pos + 1);

    if (flag.find("--layers=") != std::string::npos) {

        // Split the string following --layers= by ','
        for (auto layer : flag_value | std::views::split(',') 
                                     | std::ranges::to<std::vector<std::string>>()) {
            
            // Convert the string to "%zu", aka size_t
            size_t num_neurons = 0;

            if (!convert_flag_to_num(layer, Flag_Conversion_Type::SIZE_T, &num_neurons)) {
                return false;
            }
            // Append the number of neurons in each layer to the config
            config.layer_info.push_back(num_neurons);
        }

        // Ensure we have the minimum number of layers for a proper network
        if (config.layer_info.size() < 3) {
            Log::log_message(Log::Log_Priority::ERROR, "Model_Config::parse_flag",
                std::format("Neural_Network requires a minimum of 3 layers but only {} provided.",
                    config.layer_info.size()));
            return false;
        }

        return true;
    }
    else if (flag.find("--cost_function=") != std::string::npos) {
        
        if (flag_value == "quadratic") {
            // Use int here because Cost_Function is just an enum
            config.cost_function = 0;
        }
        else if (flag_value == "cross_entropy") {
            config.cost_function = 1;
        }
        else {
            Log::log_message(Log::Log_Priority::ERROR, "Model_Config::parse_flag",
                std::format("Invalid Cost Function provided. Options are: [quadratic, cross_entropy], but {} was provided.",
                    flag_value));
            return false;
        }

        return true;
    }
    else if (flag.find("--learning_rate=") != std::string::npos) {

        float learning_rate = 0;
        // Convert the learning_rate to a float
        if (!convert_flag_to_num(flag_value, Flag_Conversion_Type::FLOAT, &learning_rate)) {
            return false;
        }
        // Ensure we have a non-zero (and not negative) learning rate
        if (learning_rate > 0) {
            config.learning_rate = learning_rate;
            return true;
        }

        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::parse_flag",
            "Learning rate cannot be zero or negative.");
        return false;
    }
    else if (flag.find("--lambda=") != std::string::npos) {

        float lambda = 0;

        if (!convert_flag_to_num(flag_value, Flag_Conversion_Type::FLOAT, &lambda)) {
            return false;
        }

        if (lambda > 0) {
            config.lambda = lambda;
            return true;
        }

        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::parse_flag",
            "Lambda cannot be zero or negative.");
        return false;
    }
    else if (flag.find("--num_train=") != std::string::npos) {

        size_t num_train = 0;

        if (!convert_flag_to_num(flag_value, Flag_Conversion_Type::SIZE_T, &num_train)) {
            return false;
        }

        if (num_train > 0) {
            config.num_train = num_train;
            return true;
        }

        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::parse_flag",
            "num_train cannot be zero");
        return false;
    }
    else if (flag.find("--num_inference=") != std::string::npos) {

        size_t num_inference = 0;

        if (!convert_flag_to_num(flag_value, Flag_Conversion_Type::SIZE_T, &num_inference)) {
            return false;
        }

        if (num_inference > 0) {
            config.num_inference = num_inference;
            return true;
        }

        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::parse_flag",
            "num_inference cannot be zero");
        return false;
    }
    else if (flag.find("--epochs=") != std::string::npos) {

        size_t epochs = 0;

        if (!convert_flag_to_num(flag_value, Flag_Conversion_Type::SIZE_T, &epochs)) {
            return false;
        }

        if (epochs > 0) {
            config.epochs = epochs;
            return true;
        }

        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::parse_flag",
            "Epochs cannot be zero");
        return false;
    }
    else if (flag.find("--batch_size=") != std::string::npos) {

        size_t batch_size = 0;

        if (!convert_flag_to_num(flag_value, Flag_Conversion_Type::SIZE_T, &batch_size)) {
            return false;
        }

        if (batch_size > 0) {
            config.batch_size = batch_size;
            return true;
        }

        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::parse_flag",
            "Batch size cannot be zero");
        return false;
    }
    else if (flag.find("--dataset_path=") != std::string::npos) {

        config.dataset_path = flag_value;
    }
    else if (flag.find("--labels_path=") != std::string::npos) {

        config.labels_path = flag_value;
    }
    else if (flag.find("--model_path=") != std::string::npos) {
        
        config.model_path = flag_value;
    }
    return false;
}

bool Model_Config_NS::convert_flag_to_num(const std::string& flag, Flag_Conversion_Type type, void* dest) {

    if (type == Flag_Conversion_Type::FLOAT) {

        if (sscanf(flag.c_str(), FLOAT_SSCANF_FMT, (float*)dest) == 1) {
            return true;
        }

        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::convert_flag_to_num",
            std::format("Unable to convert \"{}\" to float.", flag));
        return false;
    }
    else if (type == Flag_Conversion_Type::SIZE_T) {

        if (sscanf(flag.c_str(), SIZE_T_SSCANF_FMT, (size_t*)dest) == 1) {
            return true;
        }

        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::convert_flag_to_num",
            std::format("Unable to convert \"{}\" to size_t.", flag));
        return false;
    }
    else {
        Log::log_message(Log::Log_Priority::ERROR, "Model_Config::convert_flag_to_num",
            "Invalid Flag_Conversion_Type provided.");
        return false;
    }
}

bool Model_Config_NS::check_online_training_config(Model_Config& config) {

    // Check the minimum configuration and then add default values as needed
    if (config.layer_info.size() >= 3 &&
        config.learning_rate > 0 &&
        config.lambda > 0 &&
        config.num_train > 0 &&
        !config.dataset_path.empty() &&
        !config.labels_path.empty() &&
        !config.model_path.empty()) {

        // 2 is an invalid value for cost_function (default value)
        if (config.cost_function == 2) {
            // Set the default cost_function to 0 (quadratic cost)
            config.cost_function = 0;
            Log::log_message(Log::Log_Priority::WARNING, "Model_Config::check_online_training_config",
                "No cost function provided. Defaulting to quadratic cost.");
        }
        if (config.epochs == 0) {
            // Set the number of epochs as 1 by default
            config.epochs = 1;
            Log::log_message(Log::Log_Priority::WARNING, "Model_Config::check_online_training_config",
                "No epochs specified. Defaulting to 1.");
        }
        if (config.batch_size != 0) {
            Log::log_message(Log::Log_Priority::WARNING, "Model_Config::check_online_training_config",
                std::format("Batch size set to {}, but online training does not use batch. Ignoring...", config.batch_size));
        }

        // Return true after issuing any warning related to default configs
        return true;
    }
    return false;
}

bool Model_Config_NS::check_batch_training_config(Model_Config& config) {

    // Check the minimum configuration and then add default values as needed
    if (config.layer_info.size() >= 3 &&
        config.learning_rate > 0 &&
        config.lambda > 0 &&
        config.num_train > 0 &&
        config.batch_size > 0 &&
        !config.dataset_path.empty() &&
        !config.labels_path.empty() &&
        !config.model_path.empty()) {

        // 2 is an invalid value for cost_function (default value)
        if (config.cost_function == 2) {
            // Set the default cost_function to 0 (quadratic cost)
            config.cost_function = 0;
            Log::log_message(Log::Log_Priority::WARNING, "Model_Config::check_batch_training_config",
                "No cost function provided. Defaulting to quadratic cost.");
        }
        if (config.epochs == 0) {
            // Set the number of epochs as 1 by default
            config.epochs = 1;
            Log::log_message(Log::Log_Priority::WARNING, "Model_Config::check_batch_training_config",
                "No epochs specified. Defaulting to 1.");
        }
        if (config.batch_size < 2) {
            // We need a batch size of at least 2
            Log::log_message(Log::Log_Priority::ERROR, "Model_Config::check_batch_training_config",
                "Batch size must be >= 2");
            return false;
        }

        // Return true after issuing any warning related to default configs
        return true;
    }
    return false;
}

bool Model_Config_NS::check_inference_config(Model_Config& config) {

    // Check the minimum configuration 
    if (config.num_inference > 0 &&
        !config.dataset_path.empty() &&
        !config.labels_path.empty() &&
        !config.model_path.empty()) {

        return true;
    }
    return false;
}