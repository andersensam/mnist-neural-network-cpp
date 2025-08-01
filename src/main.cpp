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
 * @version: 2025-07-23
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/main.hpp"


int main(int argc, char* argv[]) {

    Log::log_message(Log::Log_Priority::WARNING, "main::main", "Hello");

    for (int i = 1; i < argc; ++i) {
        std::cout << argv[i] << "\n";
    }

    //std::vector<size_t> layer_info = {28 * 28, 100, 10};

    //MNIST_Training_NS::train_new_model("../data/train-labels-idx1-ubyte", "../data/train-images-idx3-ubyte",
    //    layer_info, 0.1, 0.1, 3000, 1, Neural_Network_NS::Cost_Function::QUADRATIC, 
    //    "../models/test.model");

    return 0;
}
