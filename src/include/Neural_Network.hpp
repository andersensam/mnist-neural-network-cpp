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
 * TODO: Continue adding functionality 
 */

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#define NEURAL_NETWORK_DEBUG 1
#define NEURAL_NETWORK_SHOW_STEP_LOSS 1
#define NEURAL_NETWORK_SHOW_LOSS_NUM_STEPS 100

/* Markers to help with loading / saving Neural Networks */
#define NN_HEADER_MAGIC 0x0000AA00
#define NN_WEIGHTS_MAGIC 0x00000A00
#define NN_WEIGHT_BEGIN 0x00000A01
#define NN_WEIGHT_END 0x00000A02
#define NN_BIASES_MAGIC 0x00000F00
#define NN_BIAS_BEGIN 0x00000F01
#define NN_BIAS_END 0x00000F02

/**
 * File structure for model
 * 
 * uint32_t NN_HEADER_MAGIC
 * float learning_rate (generally 0.1)
 * float lambda (generally 0.1)
 * uint32_t cost_function (0 = Quadratic, 1 = Cross Entropy)
 * size_t number_of_layers
 * size_t[number_of_layers] number_of_neurons
 * 
 * uint32_t NN_WEIGHTS_MAGIC
 *   uint32_t NN_WEIGHT_BEGIN
 *   float[number_of_neurons * previous_layer_neurons] weight
 *   uint32_t NN_WEIGHT_END
 * 
 * uint32_t NN_BIASES_MAGIC
 *   uint32_t NN_BIAS_BEGIN
 *   float[number_of_neurons] bias
 *   uint32_t NN_BIAS_END
 */

/* Standard dependencies */
#include <vector>
#include <math.h>
#include <stdio.h>

/* Local dependencies */
#include "Log.hpp"
#include "Matrix.hpp"
#include "Neural_Network_Layer.hpp"

/* Using */
using Matrix = Matrix_NS::Matrix<float>;
using Neural_Network_Layer = Neural_Network_Layer_NS::Neural_Network_Layer;
using Layer_Type = Neural_Network_Layer_NS::Layer_Type;

namespace Neural_Network_NS {

typedef enum {
    QUADRATIC = 0,
    CROSS_ENTROPY = 1
} Cost_Function;

class Quadratic_Cost {
public:
    /**
     * Calculate the cost incurred by a layer
     * @param output The output from the layer
     * @param expected The expected result from the layer
     * @returns Returns a float representing the cost / loss 
     */
    static float cost(const Matrix& output, const Matrix& expected);

    /**
     * Calculate the delta between the output layer and the label
     * @param z The z layer Matrix, before the activation function is applied
     * @param output The activations from the output layer
     * @param label The correct / expected value
     * @param destination Reference to a Matrix that stores the delta
     */
    static void delta(const Matrix& z, const Matrix& output, const Matrix& label, Matrix& destination);
};

class Cross_Entropy_Cost {
public:
    /**
     * Calculate the cost incurred by a layer
     * @param output The output from the layer
     * @param expected The expected result from the layer
     * @returns Returns a float representing the cost / loss 
     */
    static float cost(const Matrix& output, const Matrix& expected);

    /**
     * Calculate the delta between the output layer and the label
     * @param z The z layer Matrix, before the activation function is applied
     * @param output The activations from the output layer
     * @param label The correct / expected value
     * @param destination Reference to a Matrix that stores the delta
     */
    static void delta(const Matrix& z, const Matrix& output, const Matrix& label, Matrix& destination);
};

class Neural_Network {
private:
    /* Private data elements */
    size_t m_num_layers = 0;
    Neural_Network_Layer** m_layers = NULL;
    float m_learning_rate = 0;
    float m_lambda = 0;
    
    /* Cost function details*/
    Cost_Function m_cost_type = Cost_Function::QUADRATIC;
    float (*cost)(const Matrix& output, const Matrix& expected) = Quadratic_Cost::cost;
    void (*delta)(const Matrix& z, const Matrix& output, const Matrix& label, Matrix& destination) = Quadratic_Cost::delta;

    /* Private functions */
    
    /**
     * Run inference and don't produce a result, storing in the output layer instead
     * @param input Reference to a Matrix to use as the input
     */
    void training_inference(const Matrix& input);
public:
    /* Public functions */

    /**
     * Constructor for Neural_Network
     * @param layer_info Vector of size_t containing the sizes of each layer and number of layers
     * @param learning_rate Hyperparameter controlling the learning rate of the network
     * @param lambda Regularization hyperparameter
     * @param cost_function Type of cost function to use
     */
    Neural_Network(const std::vector<size_t>& layer_info, float learning_rate, float lambda, Cost_Function cost_function);

    /**
     * Constructor for making a copy of a Neural_Network
     */
    Neural_Network();

    /**
     * Constructor for loading a Neural_Network from a file
     * @param path Path to the Neural_Network file
     */

    /**
     * Destructor for Neural_Network
     */
    ~Neural_Network();

    /**
     * Execute training of the Neural Network, running a single training step for one image
     * @param input The input Matrix. This should be of size [num_neurons x 1], with num_neurons
     * representing the first (input) layer
     * @param label A Matrix instance containing a single value set, this should be the same
     * size as the number of neurons in the final layer [num_labels x 1]
     * @param dataset_size The size of the full dataset
     * @returns Returns the total loss for the training step
     */
    float train(const Matrix& input, const Matrix& label, size_t dataset_size);

    /**
     * Execute batch training on the Neural Network
     * @param inputs A Matrix instance containing one input per column. The size of the Matrix
     * should be [input_neurons x batch_size]
     * @param labels A Matrix instance containing one label per column. The size of the Matrix
     * should be [num_labels x batch_size]
     * @param dataset_size The size of the full dataset
     * @returns Returns the total loss across the number of steps in the batch
     */
    float batch_train(const Matrix& inputs, const Matrix& labels, size_t dataset_size);

    /**
     * Run inference using a trained Neural Network
     * @param input A Matrix instance containing one or more inputs. Each input should
     * be in a separate column
     * @returns Returns a Matrix containing predictions for each input
     */
    Matrix* inference(const Matrix& input) const;

    /**
     * Run inference using a trained Neural Network, putting its result into a defined Matrix
     * @param input A Matrix instance containing one or more inputs. Each input should
     * be in a separate column
     * @param destination Matrix to write the results to. This Matrix should be of dimensions
     * [labels x inputs.size()]
     */
    void inference(const Matrix& input, Matrix& destination) const;

    /**
     * Create a deep copy of a Neural Network
     */
    Neural_Network* clone(void);

    /**
     * Save a Neural Network to a file
     * @param path Path to save the Neural Network at
     */
    void save(const char* path) const;
};

/**
 * Calculate the sigmoid of a given float
 * @param z Float to calculate the sigmoid of
 * @returns Returns a float of the sigmoid
 */
float sigmoid(float z);

/**
 * Calculate the softmax of a Matrix
 * @param target The Matrix to calculate the softmax of
 * @returns Returns a new Matrix instance with softmax applied
 */
Matrix* softmax(const Matrix& target);

/**
 * Calculate the softmax of a Matrix and store its results in another Matrix
 * @param target The Matrix to calculate the softmax of
 * @param destination The Matrix destination to write to
 */
void softmax(const Matrix& target, Matrix& destination);

/**
 * Calculate the sigmoid prime of a Matrix
 * @param target The Matrix to calculate the sigmoid prime of
 * @returns Returns a new Matrix instance with the sigmoid prime
 */
Matrix* sigmoid_prime(const Matrix& target);

/**
 * Calculate the sigmoid prime of a Matrix, storing in an existing Matrix
 * @param target The Matrix to calculate the sigmoid prime of
 * @param destination The destination Matrix to write to
 */
void sigmoid_prime(const Matrix& target, Matrix& destination);

};

#endif
