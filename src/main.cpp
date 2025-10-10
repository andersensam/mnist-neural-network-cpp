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

#include "include/main.hpp"


int main(int argc, char* argv[]) {

    if (argc < 2) {

        Log::log_message(Log::Log_Priority::ERROR, "main::main",
            "Invalid arguments provided. Exiting...");
        exit(-1);
    }

    Log::log_message(Log::Log_Priority::INFO, "main::main", "Starting up... options provided:");

    for (int i = 1; i < argc; ++i) {
        std::cout << argv[i] << "\n";
    }

    const std::string& command = std::string(argv[1]);
    Model_Config_NS::Model_Config config = {};

    if (command == "online-training") {
        Log::log_message(Log::Log_Priority::INFO, "main::main", "Online training mode selected. Checking trainer config...");
        for (int i = 2; i < argc; ++i) {
            Model_Config_NS::parse_flag(argv[i], config);
        }
        if (!Model_Config_NS::check_online_training_config(config)) {
            Log::log_message(Log::Log_Priority::ERROR, "main::main", "Trainer config incomplete. Please provide a valid config.");
            exit(EXIT_FAILURE);
        }
        Log::log_message(Log::Log_Priority::INFO, "main::main", "Trainer config validated. Starting training loop...");
        MNIST_Training_NS::train_new_model_online(config);
    }
    else if (command == "batch-training") {
        Log::log_message(Log::Log_Priority::INFO, "main::main", "Batch training mode selected. Checking trainer config...");
        for (int i = 2; i < argc; ++i) {
            Model_Config_NS::parse_flag(argv[i], config);
        }
        if (!Model_Config_NS::check_batch_training_config(config)) {
            Log::log_message(Log::Log_Priority::ERROR, "main::main", "Trainer config incomplete. Please provide a valid config.");
            exit(EXIT_FAILURE);
        }
        Log::log_message(Log::Log_Priority::INFO, "main::main", "Trainer config validated. Starting training loop...");
        MNIST_Training_NS::train_new_model_batch(config);
    }
    else if (command == "inference") {
        Log::log_message(Log::Log_Priority::INFO, "main::main", "Inference mode selected. Loading model...");
        for (int i = 2; i < argc; ++i) {
            Model_Config_NS::parse_flag(argv[i], config);
        }
        if (!Model_Config_NS::check_inference_config(config)) {
            Log::log_message(Log::Log_Priority::ERROR, "main::main", "Inference config incomplete. Please provide a valid config.");
            exit(EXIT_FAILURE);
        }
        Log::log_message(Log::Log_Priority::INFO, "main::main", "Inference config validated. Starting inference loop...");
        MNIST_Inference_NS::inference(config);
    }
    else if (command == "help") {
        std::cout << "Help\n";
    }
    else {
        std::cerr << "Invalid command provided\n";
    }

    return 0;
}
