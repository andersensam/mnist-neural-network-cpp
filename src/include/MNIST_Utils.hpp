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
 * TODO: Rewrite MNIST_Utils to be more C++-like. These utils are currently
 * very C-like in style
 */

#ifndef MNIST_UTILS_HPP
#define MNIST_UTILS_HPP

/* General config */
#define MNIST_UTILS_DEBUG 1

/* MNIST_Lables config */
#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_LABELS 10

/* MNIST_Images config */
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT

/* Standard dependencies */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Local dependencies */
#include "Matrix.hpp"
#include "Log.hpp"

/* Using */
using Matrix = Matrix_NS::Matrix<float>;

namespace MNIST_Utils_NS {

class MNIST_Images {
private:
    /* Private data elements */
    size_t m_num_images = 0;
    Matrix** m_images = NULL;

    /* Private functions */

    /**
     * Check to see that a given index exists and is valid (not NULL)
     * @param index Index to check
     * @returns True if it is valid, false otherwise
     */
    bool exists(size_t index) const;

public:
    /* Public functions */

    /**
     * Constructor for MNIST_Images
     * @param path Path to file containing the MNIST image collection
     */
    MNIST_Images(const char* path);

    /**
     * Destructor for MNIST_Images
     */
    ~MNIST_Images();

    /**
     * Get the number of images contained in MNIST_Images
     * @returns Returns a size_t of the number of images contained
     */
    size_t size(void) const;

    /**
     * Get the Matrix representing the image's pixels at an index
     * @param index The number of the image / Matrix we want to fetch
     * @returns Returns a constant reference to the underlying image data
     */
    const Matrix& get(size_t index) const;

    /**
     * Get a flattened Matrix representing the image pixels
     * @param index The index of the image we want to retrieve
     * @returns Returns a pointer to a new Matrix with the flattened image
     */
    Matrix* get_flat(size_t index) const;

    /**
     * Get a flattened Matrix representing the image pixels, storing in a preexisting Matrix
     * @param index The index of the image we want to retrieve
     * @param destination Reference to a Matrix we want to write the result to
     */
    void get_flat(size_t index, Matrix& destination) const;

    /**
     * Create a Matrix of multiple images combined together, useful for
     * batch training
     * @param image_start Start index 
     * @param image_end End index
     * @returns Returns a pointer to a Matrix of size MNIST_IMAGE_SIZE x (image_end - image_start)
     */
    Matrix* create_images_from_range(size_t image_start, size_t image_end) const;

    /**
     * Create a Matrix of multiple images combined together, useful for
     * batch training, storing the result in an existing Matrix
     * @param image_start Start index 
     * @param image_end End index
     * @param destination Reference to a Matrix to store the result in
     */
    void create_images_from_range(size_t image_start, size_t image_end, Matrix& destination) const;
};

class MNIST_Labels {
private:
    /* Private data elements */
    size_t m_num_labels = 0;
    uint8_t* m_labels = NULL;

    /* Private functions */

    /**
     * Check to see that a given index exists and is valid (not NULL)
     * @param index Index to check
     * @returns True if it is valid, false otherwise
     */
    bool exists(size_t index) const;

public:
    /* Public functions */

    /**
     * Constructor for MNIST_Labels
     * @param path Path to the file containing the MNIST labels
     */
    MNIST_Labels(const char* path);

    /**
     * Destructor for MNIST_Labels
     */
    ~MNIST_Labels();

    /**
     * Get the number of labels contained in MNIST_Labels
     * @returns Returns a size_t with the number of labels
     */
    size_t size(void) const;

    /**
     * Get a label at an index
     * @param index Index to retrieve
     * @returns Returns the uint8_t label at the index
     */
    uint8_t get(size_t index) const;

    /**
     * Create a Matrix representation of an MNIST label
     * @param index The index of label to create the Matrix for
     * @returns Returns a pointer to a Matrix with the label
     */
    Matrix* create_label(size_t index) const;

    /**
     * Create a Matrix representation of an MNIST label, storing in an existing Matrix
     * @param index The index of the label to crete the Matrix for
     * @param destination The Matrix to write the result to
     */
    void create_label(size_t index, Matrix& destination) const;

    /**
     * Create a Matrix representation of multiple labels, useful for batch training
     * @param label_start Start index to create labels from
     * @param label_end End index
     * @returns Returns a pointer to a new Matrix of size MNIST_LABELS x (label_end - label_start)
     */
    Matrix* create_labels_from_range(size_t label_start, size_t label_end) const;

    /**
     * Create a Matrix representation of multiple labels, storing in an existing Matrix
     * useful for batch training
     * @param label_start Start index to create labels from
     * @param label_end End index
     * @param destination The Matrix to write the result to
     */
    void create_labels_from_range(size_t label_start, size_t label_end, Matrix& destination) const;
};

/**
 * Convert uint8_t to a float representing pixel intensity
 * @param pixel A pointer to the uint8_t pixel representation to convert
 * @returns Returns a float from 0 to 1 representing pixel / 255
 */
float pixel_to_float(const uint8_t* pixel);

/**
 * Convert from the big endian format in the dataset if we're on a little endian machine
 * @param in Number that we're taking in and examining the byte order
 * @returns uint32_t in the proper endianness
 */
uint32_t map_uint32(uint32_t in);

};

#endif
