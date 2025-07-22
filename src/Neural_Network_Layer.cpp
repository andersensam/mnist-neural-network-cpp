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
 * @version: 2025-07-16
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/Neural_Network_Layer.hpp"
using Neural_Network_Layer_NS::Neural_Network_Layer;
using Neural_Network_Layer_NS::Layer_Type;

const Matrix* Neural_Network_Layer::get_matrix(Layer_Type layer_type) const {

    if (layer_type == Layer_Type::WEIGHTS) { return (const Matrix*)m_weights; }
    else if (layer_type == Layer_Type::BIASES) { return (const Matrix*)m_biases; }
    else if (layer_type == Layer_Type::OUTPUTS) { return (const Matrix*)m_outputs; }
    else if (layer_type == Layer_Type::ERRORS) { return (const Matrix*)m_errors; }
    else if (layer_type == Layer_Type::NEW_WEIGHTS) { return (const Matrix*)m_new_weights; }
    else if (layer_type == Layer_Type::Z) { return (const Matrix*)m_z; }

    Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::get_matrix",
        "Invalid Layer_Type provided. Returning NULL");

    return (const Matrix*)NULL;
}

Matrix* Neural_Network_Layer::get_matrix(Layer_Type layer_type) {

    if (layer_type == Layer_Type::WEIGHTS) { return m_weights; }
    else if (layer_type == Layer_Type::BIASES) { return m_biases; }
    else if (layer_type == Layer_Type::OUTPUTS) { return m_outputs; }
    else if (layer_type == Layer_Type::ERRORS) { return m_errors; }
    else if (layer_type == Layer_Type::NEW_WEIGHTS) { return m_new_weights; }
    else if (layer_type == Layer_Type::Z) { return m_z; }

    Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::get_matrix",
        "Invalid Layer_Type provided. Returning NULL");

    return NULL;
}

Neural_Network_Layer::Neural_Network_Layer(size_t num_neurons, size_t previous_layer_neurons, 
    bool generate_biases, bool import) {

    // Persist the number of neurons
    m_num_neurons = num_neurons;
    m_previous_layer_neurons = previous_layer_neurons;

    // Don't initialize any of the Matrix instances if this is the first (input) layer
    // or if we are using this as an import
    if (previous_layer_neurons == 0 || import) {
        return; 
    }

    // Initialize the weights, biases, etc.
    m_weights = new Matrix(num_neurons, previous_layer_neurons);

    // The bias Matrix is alawys one column wide
    m_biases = new Matrix(num_neurons, 1);

    for (size_t i = 0; i < num_neurons; ++i) {

        // If we want to generate biases, grab random floats from [-1, 1]
        if (generate_biases) {
            m_biases->set(i, 0, random_float());
        }
        else {
            m_biases->set(i, 0, 0);
        }

        // Fill the weights Matrix with random values to start
        for (size_t j = 0; j < previous_layer_neurons; ++j) {
            m_weights->set(i, j, random_float());
        }
    }
}

Neural_Network_Layer::~Neural_Network_Layer() {

    if (m_weights != NULL) { delete m_weights; }
    if (m_biases != NULL) { delete m_biases; }
    if (m_outputs != NULL) { delete m_outputs; }
    if (m_errors != NULL) { delete m_errors; }
    if (m_new_weights != NULL) { delete m_new_weights; }
    if (m_z != NULL) { delete m_z; }
}

size_t Neural_Network_Layer::get_num_neurons(void) const {

    return m_num_neurons;
}

size_t Neural_Network_Layer::get_previous_layer_num_neurons(void) const {

    return m_previous_layer_neurons;
}

Neural_Network_Layer* Neural_Network_Layer::clone(void) const {

    // Allocate a new Neural_Network_Layer using the same number of neurons
    // Set previous_layer_neurons = 0, generate_biases = false, and import = true
    // since we are going to just copy whatever elements are present here
    Neural_Network_Layer* target = new Neural_Network_Layer(m_num_neurons, m_previous_layer_neurons, false, true);

    // For each underlying Matrix, create a deep copy if it is not NULL
    if (m_weights != NULL) { target->m_weights = m_weights->clone(); }
    if (m_biases != NULL) { target->m_biases = m_biases->clone(); }
    if (m_outputs != NULL) { target->m_outputs = m_outputs->clone(); }
    if (m_errors != NULL) { target->m_errors = m_errors->clone(); }
    if (m_new_weights != NULL) { target->m_new_weights = m_new_weights->clone(); }
    if (m_z != NULL) { target->m_z = m_z->clone(); }

    return target;
}

void Neural_Network_Layer::clone(Neural_Network_Layer& destination) const {

    // Check to see that the underlying data structures are the same size
    // This should be true if the num_neurons and previous_layer_neurons are the same
    if (m_num_neurons != destination.get_num_neurons() 
        || m_previous_layer_neurons != destination.get_previous_layer_num_neurons()) {

        Log::log_message(Log::Log_Priority::ERROR, "Neural_Network_Layer::clone",
            std::format("Unable to copy Neural_Network_Layer. Caller is [{} x {}], but destination is [{} x {}]",
                m_num_neurons, m_previous_layer_neurons, destination.get_num_neurons(),
                destination.get_previous_layer_num_neurons()));

        exit(EXIT_FAILURE);
    }

    // Iterate through each Matrix, letting write_matrix handle size differences or NULL pointers
    if (m_weights != NULL) { destination.write_matrix(*m_weights, Layer_Type::WEIGHTS); }
    if (m_biases != NULL) { destination.write_matrix(*m_biases, Layer_Type::BIASES); }
    if (m_outputs != NULL) { destination.write_matrix(*m_outputs, Layer_Type::OUTPUTS); }
    if (m_errors != NULL) { destination.write_matrix(*m_errors, Layer_Type::ERRORS); }
    if (m_new_weights != NULL) { destination.write_matrix(*m_new_weights, Layer_Type::NEW_WEIGHTS); }
    if (m_z != NULL) { destination.write_matrix(*m_z, Layer_Type::Z); }
}

bool Neural_Network_Layer::exists(Layer_Type layer_type) const {

    if (layer_type == Layer_Type::WEIGHTS) {
        if (m_weights != NULL) { return true; } else {return false; }
    }
    else if (layer_type == Layer_Type::BIASES) {
        if (m_biases != NULL) { return true; } else { return false; }
    }
    else if (layer_type == Layer_Type::OUTPUTS) {
        if (m_outputs != NULL) { return true; } else { return false; }
    }
    else if (layer_type == Layer_Type::ERRORS) {
        if (m_errors != NULL) { return true; } else { return false; }
    }
    else if (layer_type == Layer_Type::NEW_WEIGHTS) {
        if (m_new_weights != NULL) { return true; } else { return false; }
    }
    else if (layer_type == Layer_Type::Z) {
        if (m_z != NULL) { return true; } else { return false; }
    }
    
    // If we didn't match any of the defined Layer_Type
    Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::exists",
        "Invalid Layer_Type provided. Returning false.");

    return false;
}

Matrix& Neural_Network_Layer::get_mutable(Layer_Type layer_type) {

    // get_matrix can return NULL if the Matrix isn't found or 
    // hasn't yet been allocated for some reason
    Matrix* target = get_matrix(layer_type);

    // If not null, dereference the pointer to return as a reference
    if (target != NULL) { return *target; }
    
    Log::log_message(Log::Log_Priority::ERROR, "Neural_Network_Layer::get_mutable",
        "NULL pointer returned in get_mutable. Cannot return a reference. Exiting");
    
    exit(EXIT_FAILURE);
}

const Matrix& Neural_Network_Layer::get_const(Layer_Type layer_type) const {

    // See notes for get_mutable for how get_matrix works
    const Matrix* target = get_matrix(layer_type);

    if (target != NULL) { return *target; }

    Log::log_message(Log::Log_Priority::ERROR, "Neural_Network_Layer::get_const",
        std::format("NULL pointer returned in get_const. Cannot return a reference for type {}. Exiting",
        (int)layer_type));
    
    exit(EXIT_FAILURE);
}

void Neural_Network_Layer::write_matrix(const Matrix& target, Layer_Type layer_type) {
    
    if (layer_type == Layer_Type::WEIGHTS) {
        // Check to see if a Matrix is already allocated
        if (exists(layer_type)) {
            // Ensure the underlying Matrix instances have the same size
            if (m_weights->size() == target.size()) {
                target.copy_to(*m_weights);
                return;
            }
            // If the sizes are not the same, free the existing Matrix
            if (NEURAL_NETWORK_LAYER_RESIZE_WARNING) {
                Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::write_matrix",
                    "Destination Matrix is of a different size. Deallocating first before cloning");
            }
            delete m_weights;
        }
        // If the Matrix does not exist / is already free (above), clone target
        m_weights = target.clone();
        return;
    }
    else if (layer_type == Layer_Type::BIASES) {
        if (exists(layer_type)) {
            if (m_biases->size() == target.size()) {
                target.copy_to(*m_biases);
                return;
            }
            if (NEURAL_NETWORK_LAYER_RESIZE_WARNING) {
                Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::write_matrix",
                    "Destination Matrix is of a different size. Deallocating first before cloning");
            }
            delete m_biases;
        }
        m_biases = target.clone();
        return;
    }
    else if (layer_type == Layer_Type::OUTPUTS) {
        if (exists(layer_type)) {
            if (m_outputs->size() == target.size()) {
                target.copy_to(*m_outputs);
                return;
            }
            if (NEURAL_NETWORK_LAYER_RESIZE_WARNING) {
                Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::write_matrix",
                    "Destination Matrix is of a different size. Deallocating first before cloning");
            }
            delete m_outputs;
        }
        m_outputs = target.clone();
        return;
    }
    else if (layer_type == Layer_Type::ERRORS) {
        if (exists(layer_type)) {
            if (m_errors->size() == target.size()) {
                target.copy_to(*m_errors);
                return;
            }
            if (NEURAL_NETWORK_LAYER_RESIZE_WARNING) {
                Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::write_matrix",
                    "Destination Matrix is of a different size. Deallocating first before cloning");
            }
            delete m_errors;
        }
        m_errors = target.clone();
        return;
    }
    else if (layer_type == Layer_Type::NEW_WEIGHTS) {
        if (exists(layer_type)) {
            if (m_new_weights->size() == target.size()) {
                target.copy_to(*m_new_weights);
                return;
            }
            if (NEURAL_NETWORK_LAYER_RESIZE_WARNING) {
                Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::write_matrix",
                    "Destination Matrix is of a different size. Deallocating first before cloning");
            }
            delete m_new_weights;
        }
        m_new_weights = target.clone();
        return;
    }
    else if (layer_type == Layer_Type::Z) {
        if (exists(layer_type)) {
            if (m_z->size() == target.size()) {
                target.copy_to(*m_z);
                return;
            }
            if (NEURAL_NETWORK_LAYER_RESIZE_WARNING) {
                Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::write_matrix",
                    "Destination Matrix is of a different size. Deallocating first before cloning");
            }
            delete m_z;
        }
        m_z = target.clone();
        return;
    }

    // Handle receiving an invalid Layer_Type
    Log::log_message(Log::Log_Priority::WARNING, "Neural_Network_Layer::write_matrix",
        "Invalid Layer_Type provided. Not doing anything");
}

void Neural_Network_Layer::expand_bias(size_t batch_size) {

    if (!exists(Layer_Type::BIASES)) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Nework_Layer::expand_bias",
            "Bias Matrix does not exist. Exiting");
        exit(EXIT_FAILURE);
    }

    const Matrix& old_bias = get_const(Layer_Type::BIASES);
    Matrix new_bias = Matrix(old_bias.rows(), batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < old_bias.rows(); ++j) {
            new_bias.set(j, i, old_bias.get(j, 0));
        }
    }

    write_matrix(new_bias, Layer_Type::BIASES);
}

void Neural_Network_Layer::shrink_bias(void) {

    if (!exists(Layer_Type::BIASES)) {
        Log::log_message(Log::Log_Priority::ERROR, "Neural_Nework_Layer::expand_bias",
            "Bias Matrix does not exist. Exiting");
        exit(EXIT_FAILURE);
    }

    const Matrix& old_bias = get_const(Layer_Type::BIASES);
    Matrix new_bias = Matrix(old_bias.rows(), 1);
    old_bias.get_column(0, new_bias);

    write_matrix(new_bias, Layer_Type::BIASES);
}

float Neural_Network_Layer_NS::random_float(void) {
    return (float) (2 * (std::rand() / (float)(RAND_MAX))) - 1.0;
}
