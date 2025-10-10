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

#include "include/MNIST_Inference.hpp"

using Matrix = Matrix_NS::Matrix<float>;
using Neural_Network = Neural_Network_NS::Neural_Network;
using MNIST_Images = MNIST_Utils_NS::MNIST_Images;
using MNIST_Labels = MNIST_Utils_NS::MNIST_Labels;

void MNIST_Inference_NS::inference(const Model_Config_NS::Model_Config& config) {

    MNIST_Images images = MNIST_Images(config.dataset_path);
    MNIST_Labels labels = MNIST_Labels(config.labels_path);
    Neural_Network nn = Neural_Network(config.model_path.c_str());

    // Setup Matrix instances that will be reused for processing images
    Matrix current_image = Matrix(MNIST_IMAGE_SIZE, 1);
    Matrix current_prediction = Matrix(MNIST_LABELS, 1);

    size_t num_correct = 0;

    for (size_t i = 0; i < config.num_inference; ++i) {

        images.get_flat(i, current_image);
        nn.inference(current_image, current_prediction);

        if (current_prediction.max_idx(Matrix_NS::Vector_Orientation::COLUMN, 0) == labels.get(i)) {
            ++num_correct;
        }
    }

    Log::log_message(Log::Log_Priority::INFO, "MNIST_Inference::inference", 
        std::format("Inference results: [{} / {}] correct. {}% accuracy rate.", 
            num_correct, config.num_inference, 100.0f * (static_cast<float>(num_correct) / static_cast<float>(config.num_inference))));
}
