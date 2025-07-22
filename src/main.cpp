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
 * @version: 2025-07-15
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/main.hpp"
using Matrix = Matrix_NS::Matrix<float>;
using Neural_Network_Layer = Neural_Network_Layer_NS::Neural_Network_Layer;
using Neural_Network = Neural_Network_NS::Neural_Network;
using Cost_Function = Neural_Network_NS::Cost_Function;
using MNIST_Images = MNIST_Utils_NS::MNIST_Images;
using MNIST_Labels = MNIST_Utils_NS::MNIST_Labels;

int main(int argc, char* argv[]) {

    Log::log_message(Log::Log_Priority::WARNING, "main::main", "Hello");

    auto layer_info = std::vector<size_t>({28 * 28, 100, 50, 10});
    Neural_Network nn = Neural_Network(layer_info, 0.15, 0.1, false, Cost_Function::CROSS_ENTROPY);
    Neural_Network* mm = nn.clone();

    MNIST_Images images = MNIST_Images("../data/train-images-idx3-ubyte");
    MNIST_Labels labels = MNIST_Labels("../data/train-labels-idx1-ubyte");

    Matrix current_image = Matrix(28 * 28, 1);
    Matrix current_label = Matrix(10, 1);

    for (size_t i = 0; i < 5000; ++i) {

        images.get_flat(i, current_image);
        labels.create_label(i, current_label);

        float loss = nn.train(current_image, current_label, images.size());

        if (i % 200 == 0) {
            Log::log_message(Log::Log_Priority::INFO, "main",
                std::format("Trainer step {} loss={}", i, loss));
        }

        //if (i == 5) {exit(EXIT_SUCCESS);}
    }

    Matrix current_images = Matrix(28 * 28, 5);
    Matrix current_labels = Matrix(10, 5);

    for (size_t i = 0; i < 10000; i += 5) {

        images.create_images_from_range(i, i + 5, current_images);
        labels.create_labels_from_range(i, i + 5, current_labels);

        float loss = mm->batch_train(current_images, current_labels, images.size());

        if (i % 200 == 0) {
            Log::log_message(Log::Log_Priority::INFO, "main",
                std::format("Batch trainer step {} loss={}", i, loss));
        }
    }

    MNIST_Images val_images = MNIST_Images("../data/t10k-images-idx3-ubyte");
    MNIST_Labels val_labels = MNIST_Labels("../data/t10k-labels-idx1-ubyte");

    Matrix result_nn = Matrix(10, 1);
    Matrix result_mm = Matrix(10, 1);

    size_t correct_nn = 0;
    size_t correct_mm = 0;

    for (size_t i = 0; i < 100; ++i) {

        images.get_flat(i, current_image);
        nn.inference(current_image, result_nn);
        mm->inference(current_image, result_mm);

        if (result_nn.max_idx(Matrix_NS::Vector_Orientation::COLUMN, 0) == (size_t)labels.get(i)) {
            ++correct_nn;
        }

        if (result_mm.max_idx(Matrix_NS::Vector_Orientation::COLUMN, 0) == (size_t)labels.get(i)) {
            ++correct_mm;
        }

    }

    Log::log_message(Log::Log_Priority::INFO, "main",
        std::format("NN correct {}/100. MM correct {}/100", correct_nn, correct_mm));

    delete mm;

    return 0;
}
