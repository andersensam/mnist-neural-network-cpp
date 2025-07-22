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

#include "include/Neural_Network.hpp"
using Neural_Network_NS::Neural_Network;
using Neural_Network_NS::Quadratic_Cost;
using Neural_Network_NS::Cross_Entropy_Cost;

float Quadratic_Cost::cost(const Matrix& output, const Matrix& expected) {

    Matrix error = Matrix(output.rows(), output.cols());
    // Subtract the output from the expected value, storing in error
    expected.subtract(output, error);
    // Square the difference
    error.apply_second_o(pow, 2);
    // Get the sum of distances and take square root
    float distance = sqrtf(error.sum());
    // Return 1/2 * (distance ^ 2)
    return 0.5f * pow(distance, 2);
}

void Quadratic_Cost::delta(const Matrix& z, const Matrix& output, const Matrix& label, Matrix& destination) {

    if (destination.rows() != output.rows() || destination.cols() != output.cols()) {
        Log::log_message(Log::Log_Priority::ERROR, "Quadratic_Cost::delta",
            "Destination Matrix does not match output Matrix.");
        exit(EXIT_FAILURE);
    }

    // Subtract the output from the label, storing in destination
    label.subtract(output, destination);

    // Create a Matrix to calculate the sigmoid prime
    Matrix sp = Matrix(output.rows(), output.cols());
    sigmoid_prime(z, sp);

    // Multiply the difference between (label - output) * sigmoid_prime
    destination.multiply_o(sp);
}

float Cross_Entropy_Cost::cost(const Matrix& output, const Matrix& expected) {

    Matrix output_log = Matrix(output.rows(), output.cols());
    output.apply(log10, output_log);
    output_log.multiply_o(expected);

    return (-1.0f) * output_log.sum();
}

void Cross_Entropy_Cost::delta(const Matrix& z, const Matrix& output, const Matrix& label, Matrix& destination) {

    if (destination.rows() != output.rows() || destination.cols() != output.cols()) {
        Log::log_message(Log::Log_Priority::ERROR, "Cross_Entropy_Cost::delta",
            "Destination Matrix does not match output Matrix.");
        exit(EXIT_FAILURE);
    }

    // Subtract the output from the label, storing in destination
    label.subtract(output, destination);
}

void Neural_Network::training_inference(const Matrix& input) {

    // Copy the the inputs to the first layer's outputs
    m_layers[0]->write_matrix(input, Layer_Type::OUTPUTS);

    // Begin feed-forward
    for (size_t i = 1; i < m_num_layers; ++i) {
        // Special processing for the first hidden layer
        if (i == 1) {
            Matrix hidden_inputs = Matrix(m_layers[i]->get_const(Layer_Type::WEIGHTS).rows(),
                input.cols());

            // Dot product of the first hidden layer's weights by the image input
            m_layers[i]->get_const(Layer_Type::WEIGHTS).dot(input, hidden_inputs);

            // Add the bias before proceeding
            hidden_inputs.add_o(m_layers[i]->get_const(Layer_Type::BIASES));

            // Save the output before applying the activation function
            m_layers[i]->write_matrix(hidden_inputs, Layer_Type::Z);

            // The output of this layer is the input with the activation function applied
            hidden_inputs.apply_o(sigmoid);

            // Copy the activation outputs and store in the layer for later use
            m_layers[i]->write_matrix(hidden_inputs, Layer_Type::OUTPUTS);
        }
        else {
            Matrix hidden_inputs = Matrix(m_layers[i]->get_const(Layer_Type::WEIGHTS).rows(),
                m_layers[i - 1]->get_const(Layer_Type::OUTPUTS).cols());

            // The next layer's inputs are the dot of this layer's weights by the previous layer's output
            m_layers[i]->get_const(Layer_Type::WEIGHTS).dot(
                m_layers[i - 1]->get_const(Layer_Type::OUTPUTS), hidden_inputs);
                
            hidden_inputs.add_o(m_layers[i]->get_const(Layer_Type::BIASES));
            m_layers[i]->write_matrix(hidden_inputs, Layer_Type::Z);
            hidden_inputs.apply_o(sigmoid);
            m_layers[i]->write_matrix(hidden_inputs, Layer_Type::OUTPUTS);
        }
    }
}

Neural_Network::Neural_Network(const std::vector<size_t>& layer_info, float learning_rate, float lambda, 
    bool generate_biases, Cost_Function cost_type) {

    if (layer_info.size() == 0) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::Neural_Network",
            "Cannot create a Neural Network with 0 layers");
        exit(EXIT_FAILURE);
    }

    // Persist information about the Neural Network
    m_num_layers = layer_info.size();
    m_learning_rate = learning_rate;
    m_lambda = lambda;
    
    if (cost_type == Cost_Function::CROSS_ENTROPY) {
        cost = Cross_Entropy_Cost::cost;
        delta = Cross_Entropy_Cost::delta;
    }

    m_cost_type = cost_type;
    
    // Allocate memory for the layers
    m_layers = (Neural_Network_Layer**)(calloc(m_num_layers, sizeof(Neural_Network_Layer*)));
    if (m_layers == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::Neural_Network",
            "Unable to allocate memory for layers. Exiting now...");
        exit(EXIT_FAILURE);
    }

    // Handle the first layer differently
    m_layers[0] = new Neural_Network_Layer(layer_info[0], 0, generate_biases, false);

    // For the remaining layers, iterate over layer_info, pulling the number of neurons and the previous
    // layer's neurons too
    for (size_t i = 1; i < m_num_layers; ++i) {
        m_layers[i] = new Neural_Network_Layer(layer_info[i], layer_info[i - 1], generate_biases, false);
    }
}

Neural_Network::Neural_Network() {}

Neural_Network::~Neural_Network() {

    // Ensure m_layers actually exists before cleaning up
    if (m_layers != NULL) {
        for (size_t i = 0; i < m_num_layers; ++i) {
            if (m_layers[i] != NULL) {
                delete m_layers[i];
                m_layers[i] = NULL;
            }
            else {
                Log::log_message(Log::Log_Priority::WARNING, "Neural_Nework::~Neural_Network",
                    std::format("Neural_Network_Layer {} is NULL. Not deallocating", i));
            }
        }
        // Use free since we allocated raw here
        free(m_layers);
        m_layers = NULL;
    }
    else {
        Log::log_message(Log::Log_Priority::WARNING, "Neural_Network::~Neural_Network",
            "m_layers is NULL. Not deallocating");
    }
}

float Neural_Network::train(const Matrix& input, const Matrix& label, size_t dataset_size) {

    if (m_layers == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::train",
            "m_layers is NULL. Cannot perform training");
        exit(EXIT_FAILURE);
    }

    // Run "inference" with the provided input
    training_inference(input);

    // Have somewhere to store the loss for a step
    float total_loss = 0;

    // Begin the backpropagation part of the training process
    for (size_t i = m_num_layers - 1; i >= 1; --i) {

        // Special processing for the output layer
        if (i == m_num_layers - 1) {
            // Calculate the delta from the predicted output and the label
            Matrix error = Matrix(label.rows(), label.cols());
            delta(m_layers[i]->get_const(Layer_Type::Z), m_layers[i]->get_const(Layer_Type::OUTPUTS),
                label, error);

            // Persist the delta as error
            m_layers[i]->write_matrix(error, Layer_Type::ERRORS);

            // Get the loss for this training step
            total_loss += cost(m_layers[i]->get_const(Layer_Type::OUTPUTS), label);
        }
        else {
            // Process the remainder of the layers differently

            // Calculate the sigmoid prime of z
            const Matrix& z = m_layers[i]->get_const(Layer_Type::Z);
            Matrix sp = Matrix(z.rows(), z.cols());
            sigmoid_prime(z, sp);

            // Get the transpose of the next layer's weights
            const Matrix& next_weights = m_layers[i + 1]->get_const(Layer_Type::WEIGHTS);
            Matrix nw_t = Matrix(next_weights.rows(), next_weights.cols());
            next_weights.copy_to(nw_t);
            nw_t.transpose_self();

            // Get the previous layer's error
            const Matrix& prev_error = m_layers[i + 1]->get_const(Layer_Type::ERRORS);

            // Calculate the error for this layer
            Matrix error = Matrix(nw_t.rows(), prev_error.cols());
            nw_t.dot(prev_error, error);
            error.multiply_o(sp);

            // Persist the new error
            m_layers[i]->write_matrix(error, Layer_Type::ERRORS);
        }

        // Get the transpose of previous layer's output
        const Matrix& prev_output = m_layers[i - 1]->get_const(Layer_Type::OUTPUTS);
        Matrix po_t = Matrix(prev_output.rows(), prev_output.cols());
        prev_output.copy_to(po_t);
        po_t.transpose_self();

        const Matrix& error = m_layers[i]->get_const(Layer_Type::ERRORS);

        // Calculate the new weights
        Matrix new_weights = Matrix(error.rows(), po_t.cols());
        error.dot(po_t, new_weights);
        m_layers[i]->write_matrix(new_weights, Layer_Type::NEW_WEIGHTS);
    }

    // Begin updating weights and biases
    for (size_t i = 1; i < m_num_layers; ++i) {
        // Get mutable references to the weights and biases to update them
        Matrix& weights = m_layers[i]->get_mutable(Layer_Type::WEIGHTS);
        Matrix& biases = m_layers[i]->get_mutable(Layer_Type::BIASES);
        Matrix& nw = m_layers[i]->get_mutable(Layer_Type::NEW_WEIGHTS);
        Matrix& error = m_layers[i]->get_mutable(Layer_Type::ERRORS);

        // Scale down the current weights by a factor of (1 - (learning_rate * lambda dataset_size))
        weights.scale_o(1 - (m_learning_rate * (m_lambda / dataset_size)));
        // Scale the new weights by the learning rate
        nw.scale_o(m_learning_rate);
        // Add the scaled new weights to the current weights
        weights.add_o(nw);

        // Scale the new biases by the learning rate
        error.scale_o(m_learning_rate);
        // Add the scaled new biases to the current biases
        biases.add_o(error);
    }

    return total_loss;
}

float Neural_Network::batch_train(const Matrix& inputs, const Matrix& labels, size_t dataset_size) {

    if (m_layers == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::train",
            "m_layers is NULL. Cannot perform training");
        exit(EXIT_FAILURE);
    }

    if (inputs.cols() != labels.cols()) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::train",
            "Number of columns in inputs and labels needs to match");
        if (NEURAL_NETWORK_DEBUG) {
            Log::log_message(Log::Log_Priority::DEBUG, "Neural_Network::train",
                std::format("Inputs has {} columns, while labels has {}",
                    inputs.cols(), labels.cols()));
        }
        exit(EXIT_FAILURE);
    }

    // Setup our nablas, one for each layer except for the input layer
    Matrix** nabla_w = (Matrix**)calloc(m_num_layers - 1, sizeof(Matrix*));
    Matrix** nabla_b = (Matrix**)calloc(m_num_layers - 1, sizeof(Matrix*));

    if (nabla_w == NULL || nabla_b == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::train",
            "Unable to allocate memory for nabla_w or nabla_b. Exiting");
        exit(EXIT_FAILURE);
    }

    // Initialize values for the nablas
    for (size_t i = 0; i < m_num_layers - 1; ++i) {
        nabla_w[i] = NULL;
        nabla_b[i] = NULL;
    }

    // Track loss across the batch
    float total_loss = 0;
    // Have the batch size easily available
    size_t batch_size = inputs.cols();

    // Expand the bias Matrix instances to be of dimension [neurons x batch_size]
    for (size_t i = 1; i < m_num_layers; ++i) {
        m_layers[i]->expand_bias(batch_size);
    }

    // Create a Matrix of ones to do a sum operation later
    Matrix ones = Matrix(batch_size, 1);
    ones.populate(1.0);

    // Run inference on the Matrix of inputs and store their outputs in each layer
    training_inference(inputs);

    // Begin backpropagation
    for (size_t i = m_num_layers - 1; i >= 1; --i) {
        
        // The error calculation should be exactly the same as the single training example above
        // Special processing for the output layer
        if (i == m_num_layers - 1) {
            // Calculate the delta from the predicted output and the label
            Matrix error = Matrix(labels.rows(), labels.cols());
            delta(m_layers[i]->get_const(Layer_Type::Z), m_layers[i]->get_const(Layer_Type::OUTPUTS),
                labels, error);

            // Persist the delta as error
            m_layers[i]->write_matrix(error, Layer_Type::ERRORS);

            // Get the loss for this training step
            total_loss += cost(m_layers[i]->get_const(Layer_Type::OUTPUTS), labels);
        }
        else {
            // Process the remainder of the layers differently

            // Calculate the sigmoid prime of z
            const Matrix& z = m_layers[i]->get_const(Layer_Type::Z);
            Matrix sp = Matrix(z.rows(), z.cols());
            sigmoid_prime(z, sp);

            // Get the transpose of the next layer's weights
            const Matrix& next_weights = m_layers[i + 1]->get_const(Layer_Type::WEIGHTS);
            Matrix nw_t = Matrix(next_weights.rows(), next_weights.cols());
            next_weights.copy_to(nw_t);
            nw_t.transpose_self();

            // Get the previous layer's error
            const Matrix& prev_error = m_layers[i + 1]->get_const(Layer_Type::ERRORS);

            // Calculate the error for this layer
            Matrix error = Matrix(nw_t.rows(), prev_error.cols());
            nw_t.dot(prev_error, error);
            error.multiply_o(sp);

            // Persist the new error
            m_layers[i]->write_matrix(error, Layer_Type::ERRORS);
        }

        // Get the transpose of previous layer's output
        const Matrix& prev_output = m_layers[i - 1]->get_const(Layer_Type::OUTPUTS);
        Matrix po_t = Matrix(prev_output.rows(), prev_output.cols());
        prev_output.copy_to(po_t);
        po_t.transpose_self();

        const Matrix& error = m_layers[i]->get_const(Layer_Type::ERRORS);

        // Get the dot product of the transposed outputs and the errors * sigmoid
        nabla_w[i - 1] = error.dot(po_t);
        nabla_b[i - 1] = error.dot(ones);
    }

    // Begin the final calculations for the new weights

    for (size_t i = 1; i < m_num_layers; ++i) {

        // Divide the sum of the deltas per layer by the batch size and multiply by the learning rate
        nabla_w[i - 1]->scale_o(m_learning_rate / (float)batch_size);
        nabla_b[i - 1]->scale_o(m_learning_rate / (float)batch_size);

        // Add the processed changes to the original weights
        m_layers[i]->get_mutable(Layer_Type::WEIGHTS).scale_o(1 - (m_learning_rate * (m_lambda / dataset_size)));
        m_layers[i]->get_mutable(Layer_Type::WEIGHTS).add_o(*(nabla_w[i - 1]));

        // Clean up nabla_w
        delete nabla_w[i - 1];

        // Convert the bias Matrix back to being one column wide
        m_layers[i]->shrink_bias();

        // Add the processed changes to the original biases
        m_layers[i]->get_mutable(Layer_Type::BIASES).add_o(*(nabla_b[i - 1]));
        delete nabla_b[i - 1];
    }

    free(nabla_w);
    free(nabla_b);

    return total_loss / batch_size;
}

Matrix* Neural_Network::inference(const Matrix& input) const {

    if (m_layers == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::inference",
            "m_layers is NULL. Cannot perform inference");
        exit(EXIT_FAILURE);
    }

    Matrix* result = new Matrix(m_layers[m_num_layers - 1]->get_num_neurons(), input.cols());
    inference(input, *result);
    return result;
}

void Neural_Network::inference(const Matrix& input, Matrix& destination) const {

    if (m_layers == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::inference",
            "m_layers is NULL. Cannot perform inference");
        exit(EXIT_FAILURE);
    }

    if (m_layers[m_num_layers - 1]->get_num_neurons() != destination.rows()) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::inference",
            "Destination Matrix's number of rows does not match the number of neurons in output layer");
        exit(EXIT_FAILURE);
    }

    // Use m_num_layers - 1 since the first layer's output is the actual input
    Matrix** outputs = (Matrix**)calloc(m_num_layers - 1, sizeof(Matrix*));

    if (outputs == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::inference",
            "Unable to allocate memory for outputs");
        exit(EXIT_FAILURE);
    }

    // Run feed-forward
    for (size_t i = 1; i < m_num_layers; ++i) {
        if (i == 1) {
            outputs[i - 1] = m_layers[i]->get_const(Layer_Type::WEIGHTS).dot(input);
        }
        else {
            outputs[i - 1] = m_layers[i]->get_const(Layer_Type::WEIGHTS).dot(*(outputs[i - 2]));
        }

        // Add the biases to the input of the layer
        outputs[i - 1]->add_o(m_layers[i]->get_const(Layer_Type::BIASES));

        // Apply the activation function
        outputs[i - 1]->apply_o(sigmoid);
    }

    // Run softmax against each inference result (if more than one column)
    softmax(*(outputs[m_num_layers - 2]), destination);

    // Clean up the allocated Matrix instances
    for (size_t i = 0; i < m_num_layers - 1; ++i) {
        delete outputs[i];
    }
    free(outputs);
}

Neural_Network* Neural_Network::clone(void) {

    Neural_Network* target = new Neural_Network();
    target->m_num_layers = m_num_layers;
    target->m_learning_rate = m_learning_rate;
    target->m_lambda = m_lambda;
    target->m_layers = (Neural_Network_Layer**)calloc(m_num_layers, sizeof(Neural_Network_Layer*));

    if (target->m_layers == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "Nerual_Network::clone",
            "Error allocating memory for m_layers");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < m_num_layers; ++i) {
        target->m_layers[i] = m_layers[i]->clone();
    }

    target->m_cost_type = m_cost_type;

    if (m_cost_type == Cost_Function::CROSS_ENTROPY) {
        target->cost = Cross_Entropy_Cost::cost;
        target->delta = Cross_Entropy_Cost::delta;
    }

    return target;
}

float Neural_Network_NS::sigmoid(float z) {

    return 1.0f / (1 + exp(-1 * z));
}

Matrix* Neural_Network_NS::softmax(const Matrix& target) {

    Matrix* result = new Matrix(target.rows(), target.cols());
    softmax(target, *result);
    return result;
}

void Neural_Network_NS::softmax(const Matrix& target, Matrix& destination) {

    if (target.rows() != destination.rows() || target.cols() != destination.cols()) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::softmax",
            "Incorrect destination Matrix size provided");
        exit(EXIT_FAILURE);
    }

    // Copy the contents of the target to the destination
    target.copy_to(destination);
    // Apply the exp function across all elements of destination
    destination.apply_o(exp);

    for (size_t i = 0; i < target.cols(); ++i) {
        // Create a per-column total
        float total = 0;
        for (size_t j = 0; j < target.rows(); ++j) {
            total += exp(target.get(j, i));
        }
        // Set the destination Matrix values to (current * 1/total)
        for (size_t j = 0; j < target.rows(); ++j) {
            destination.set(j, i, destination.get(j, i) * (1.0f / total));
        }
    }
}

Matrix* Neural_Network_NS::sigmoid_prime(const Matrix& target) {

    Matrix* result = new Matrix(target.rows(), target.cols());
    sigmoid_prime(target, *result);
    return result;
}

void Neural_Network_NS::sigmoid_prime(const Matrix& target, Matrix& destination) {

    if (target.rows() != destination.rows() || target.cols() != destination.cols()) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network::sigmoid_prime",
            "Target and destination sizes are different. Cannot proceed");
        exit(EXIT_FAILURE);
    }
    
    // Calculate the sigmoid of the Matrix
    Matrix t_sigmoid = Matrix(target.rows(), target.cols());
    target.copy_to(t_sigmoid);
    t_sigmoid.apply_o(sigmoid);

    // Subtract the sigmoid from a Matrix of all ones
    destination.populate(1.0);
    destination.subtract_o(t_sigmoid);

    // Multiply the sigmoid by (1 - sigmoid)
    destination.multiply_o(t_sigmoid);
}
