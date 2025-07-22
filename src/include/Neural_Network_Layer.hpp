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
 * @version: 2025-07-21
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef NEURAL_NETWORK_LAYER_HPP
#define NEURAL_NETWORK_LAYER_HPP

#define NEURAL_NETWORK_LAYER_DEBUG 1
#define NEURAL_NETWORK_LAYER_RESIZE_WARNING 0

/* Standard dependencies */
#include <cstdlib>
#include <math.h>

/* Local dependencies */
#include "Log.hpp"
#include "Matrix.hpp"

/* Using */
using Matrix = Matrix_NS::Matrix<float>;

/* Definitions */
namespace Neural_Network_Layer_NS {

typedef enum {
    WEIGHTS = 0,
    BIASES = 1,
    OUTPUTS = 2,
    ERRORS = 3,
    NEW_WEIGHTS = 4,
    NEW_BIASES = 5,
    Z = 6
} Layer_Type;

class Neural_Network_Layer {
private:
    /* Private data elements */
    size_t m_num_neurons = 0;
    size_t m_previous_layer_neurons = 0;
    Matrix* m_weights = NULL;
    Matrix* m_biases = NULL;
    Matrix* m_outputs = NULL;
    Matrix* m_errors = NULL;
    Matrix* m_new_weights = NULL;
    Matrix* m_new_biases = NULL;
    Matrix* m_z = NULL;

    /* Private functions */

    /**
     * Use the Layer_Type enum to get the raw pointer to a Matrix, if it exists
     * @param layer_type Enum Layer_Type containing the Matrix we want to get
     * @returns Returns a const pointer to the Matrix, otherwise returns NULL
     */
    const Matrix* get_matrix(Layer_Type layer_type) const;

    /**
     * Use the Layer_Type enum to get the raw pointer to a Matrix, if it exists
     * @param layer_type Enum Layer_Type containing the Matrix we want to get
     * @returns Returns a pointer to the Matrix, otherwise returns NULL
     */
    Matrix* get_matrix(Layer_Type layer_type);

public:
    /* Public functions */

    /**
     * Create a new Neural_Network_Layer
     * @param num_neurons Number of neurons contained in this layer
     * @param previous_layer_neurons Number of neurons in the previous layer
     * @param generate_biases True to generate biases, fale otherwise
     * @param import True to setup empty layer and copy data into later
     * @returns Returns a new Neural_Network_Layer
     */
    Neural_Network_Layer(size_t num_neurons, size_t previous_layer_neurons, bool generate_biases, bool import);

    /**
     * Destructor for Neural_Network_Layer
     */
    ~Neural_Network_Layer();

    /**
     * Get the number of neurons present in this layer
     * @returns Returns size_t with the number of neurons
     */
    size_t get_num_neurons(void) const;

    /**
     * Get the number of neurons present in the previous layer
     * @returns Returns size_t with the number of neurons in the previous layer
     */
    size_t get_previous_layer_num_neurons(void) const;

    /**
     * Create a deep copy of a Neural_Network_Layer
     * @returns Returns a pointer to a Neural_Network_Layer copy
     */
    Neural_Network_Layer* clone(void) const;

    /**
     * Create a deep copy of a Neural_Network_Layer, storing in a predefined destination
     * @param destination Reference to a destination Neural_Network_Layer
     */
    void clone(Neural_Network_Layer& destination) const;

    /**
     * Check to see if a Matrix exists within a layer
     * @param layer_type Layer_Type enum representing the type we are checking for
     * @returns Returns true if it is a valid pointer, false if NULL
     */
    bool exists(Layer_Type layer_type) const;

    /**
     * Get a mutable reference to an underlying Matrix in the layer
     * @param layer_type Layer_Type enum representing the type we want the reference to
     * @returns Returns a reference to the underlying Matrix
     */
    Matrix& get_mutable(Layer_Type layer_type);

    /**
     * Get a const reference to an underlying Matrix in the layer
     * @param layer_type Layer_Type enum representing the type we want the reference to
     * @returns Returns a const reference to the underlying Matrix
     */
    const Matrix& get_const(Layer_Type layer_type) const;

    /**
     * Write a Matrix to the layer
     * @param target Reference to the Matrix we want to write to the layer
     * @param layer_type Layer_Type enum representing the Matrix we are referring to
     */
    void write_matrix(const Matrix& target, Layer_Type layer_type);

    /**
     * Expand the bias Matrix for use during batch training
     * @param batch_size The batch size being used -- this translates to the number of columns
     * present in the updated bias Matrix
     */
    void expand_bias(size_t batch_size);

    /**
     * Shrink the bias Matrix back to its normal use
     */
    void shrink_bias(void);
};

/**
 * Generate a random float between [-1, 1]
 * @returns Returns a random float
 */
float random_float(void);

};

#endif
