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

#include "include/main.hpp"


int main(int argc, char* argv[]) {

    if (argc < 2) {

        Log::log_message(Log::Log_Priority::ERROR, "main::main",
            "Invalid arguments provided. Exiting...");
        exit(-1);
    }

    Log::log_message(Log::Log_Priority::WARNING, "main::main", "Hello");

    for (int i = 1; i < argc; ++i) {
        std::cout << argv[i] << "\n";
    }

    const std::string& command = std::string(argv[1]);
    Model_Config_NS::Model_Config config = {};

    if (command == "train") {
        Log::log_message(Log::Log_Priority::INFO, "main::main", "Training mode selected. Checking trainer config...");
        for (int i = 2; i < argc; ++i) {
            Model_Config_NS::parse_flag(argv[i], config);
        }
        if (!Model_Config_NS::check_training_config(config)) {
            Log::log_message(Log::Log_Priority::ERROR, "main::main", "Trainer config incomplete. Please provide a valid config.");
            exit(-1);
        }
        Log::log_message(Log::Log_Priority::INFO, "main::main", "Trainer config validated. Starting training loop...");
        MNIST_Training_NS::train_new_model(config);
    }
    else if (command == "inference") {
        std::cout << "inf\n";
    }
    else if (command == "help") {
        std::cout << "Help\n";
    }
    else {
        std::cerr << "Invalid command provided\n";
    }

    return 0;
}
