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

#include "include/MNIST_Utils.hpp"

using MNIST_Utils_NS::MNIST_Images;
using MNIST_Utils_NS::MNIST_Labels;

bool MNIST_Images::exists(size_t index) const {
    if (index >= m_num_images) {
        Log::log_message(Log::Log_Priority::WARNING, "MNIST_Images::exists",
            "Invalid index provided");
        if (MNIST_UTILS_DEBUG) {
            Log::log_message(Log::Log_Priority::DEBUG, "MNIST_Images::exists",
                std::format("Index {} request, but MNIST_Images ends at index {}",
                    index, m_num_images - 1));
        }
        return false;
    }
    if (m_images == NULL) {
        Log::log_message(Log::Log_Priority::WARNING, "MNIST_Images::exists",
            "m_images is NULL");
        return false;
    }
    if (m_images[index] == NULL) {
        Log::log_message(Log::Log_Priority::WARNING, "MNIST_Images::exists",
            std::format("Requested index {} is NULL", index));
        return false;
    }
    return true;
}

MNIST_Images::MNIST_Images(const char* path) {

    // Open the path to where the images are stored, in read-only mode
    FILE* images_file = fopen(path, "ro");

    if (images_file == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::MNIST_Images",
            std::format("Unable to open path '{}' to MNIST images. Exiting...",
                path));
        exit(EXIT_FAILURE);
    }

    // We want to read in four values at once here to avoid using fread again and again
    uint32_t image_buffer[4] = {0, 0, 0, 0};

    // Read in 16 bytes from the file, grabbing the magic number and the number of items contained
    if (fread(&image_buffer, sizeof(uint32_t), 4, images_file) != 4) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::MNIST_Images",
            "Unable to read headers from MNIST images file");
        fclose(images_file);
        exit(EXIT_FAILURE);
    }

    // The first entry in the array is used for the magic number; the second is for the number of items
    if (map_uint32(image_buffer[0]) != MNIST_IMAGE_MAGIC) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::MNIST_Images",
            "Mismatched magic number in the MNIST images file header");
        fclose(images_file);
        exit(EXIT_FAILURE);
    }

    // Validate the image size matches what we are expecting
    if (map_uint32(image_buffer[2]) != MNIST_IMAGE_HEIGHT || map_uint32(image_buffer[3]) != MNIST_IMAGE_WIDTH) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::MNIST_Images",
            std::format("Unexpected image dimensions provided. Check MNIST_Utils.hpp. Detected [{} x {}] but expected [{} x {}]",
                map_uint32(image_buffer[2]), map_uint32(image_buffer[3]), MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH));
        fclose(images_file);
        exit(EXIT_FAILURE);
    }

    // Store the information regarding the images
    m_num_images = (size_t)map_uint32(image_buffer[1]);
    if (MNIST_UTILS_DEBUG) {
        Log::log_message(Log::Log_Priority::INFO, "MNIST_Images::MNIST_Images",
            std::format("Reading {} images", m_num_images));
    }

    // Allocate memory for storing the image Matrix instances
    m_images = (Matrix**)calloc(m_num_images, sizeof(Matrix*));
    if (m_images == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::MNIST_Images",
            "Unable to allocate memory for storing image data");
        fclose(images_file);
        exit(EXIT_FAILURE);
    }

    uint8_t pixel_value = 0;

    for (uint32_t i = 0; i < m_num_images; ++i) {

        // For each expected image, create a new Matrix to store it in
        m_images[i] = new Matrix(MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH);

        // For each expected pixel, grab the value and store it in the Matrix
        for (size_t j = 0; j < MNIST_IMAGE_HEIGHT; ++j) {
            for (size_t k = 0; k < MNIST_IMAGE_WIDTH; ++k) {
                if (fread(&pixel_value, sizeof(uint8_t), 1, images_file) != 1) {
                    Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::MNIST_Images",
                        "Failed to read pixel data");
                    if (MNIST_UTILS_DEBUG) {
                        Log::log_message(Log::Log_Priority::DEBUG, "MNIST_Images::MNIST_Images",
                            std::format("Failed to read pixel @ index ({}, {}, {})", i, j, k));
                    }
                    fclose(images_file);
                    exit(EXIT_FAILURE);
                }
                // Copy the value of the pixel intensity to the newly created Matrix
                m_images[i]->set(j, k, pixel_to_float(&pixel_value));
            }
        }
    }

    // Cleanup
    fclose(images_file);
}

MNIST_Images::~MNIST_Images() {

    if (m_images != NULL) {
        for (size_t i = 0; i < m_num_images; ++i) {
            if (m_images[i] != NULL) {
                delete m_images[i];
                m_images[i] = NULL;
            }
            else {
                Log::log_message(Log::Log_Priority::WARNING, "MNIST_Images::~MNIST_Images",
                    "NULL pointer detected when deallocated images. Skipping");
            }
        }
        free(m_images);
        m_images = NULL;
    }
    else {
        Log::log_message(Log::Log_Priority::WARNING, "MNIST_Images::~MNIST_Images",
            "NULL pointer detected for m_images. Skipping deallocation");
    }
}

size_t MNIST_Images::size(void) const {
    return m_num_images;
}

const Matrix& MNIST_Images::get(size_t index) const {

    if (!exists(index)) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::get",
            "Invalid index provided. Exiting");
        exit(EXIT_FAILURE);
    }

    return *(m_images[index]);
}

Matrix* MNIST_Images::get_flat(size_t index) const {

    if (!exists(index)) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::get_flat",
            "Invalid index provided. Exiting");
        exit(EXIT_FAILURE);
    }

    Matrix* result = new Matrix(MNIST_IMAGE_SIZE, 1);
    get_flat(index, *result);
    return result;
}

void MNIST_Images::get_flat(size_t index, Matrix& destination) const {

    if (!exists(index)) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::get_flat",
            "Invalid index provided. Exiting");
        exit(EXIT_FAILURE);
    }
    if (destination.rows() * destination.cols() != MNIST_IMAGE_SIZE) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::get_flat",
            "Incorrect destination Matrix size");
        exit(EXIT_FAILURE);
    }
    get(index).copy_to(destination);
    destination.flatten(Matrix_NS::Vector_Orientation::COLUMN);
}

Matrix* MNIST_Images::create_images_from_range(size_t image_start, size_t image_end) const {

    if (image_start >= m_num_images || image_end >= m_num_images) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::create_images_from_range",
           "Invalid range provided");
        if (MNIST_UTILS_DEBUG) {
            Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::create_images_from_range",
                std::format("Got range ({}, {}), but m_images ends at index {}",
                    image_start, image_end, m_num_images - 1));
        } 
        exit(EXIT_FAILURE);
    }
    if (m_images == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::create_images_from_range",
            "m_images is NULL");
        exit(EXIT_FAILURE);
    }

    // Allocate the new Matrix for the flat images
    Matrix* result = new Matrix(MNIST_IMAGE_SIZE, image_end - image_start);
    create_images_from_range(image_start, image_end, *result);
    return result;
}

void MNIST_Images::create_images_from_range(size_t image_start, size_t image_end, Matrix& destination) const {

    if (image_start >= m_num_images || image_end >= m_num_images) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::create_images_from_range",
           "Invalid range provided");
        if (MNIST_UTILS_DEBUG) {
            Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::create_images_from_range",
                std::format("Got range ({}, {}), but m_images ends at index {}",
                    image_start, image_end, m_num_images - 1));
        } 
        exit(EXIT_FAILURE);
    }
    if (m_images == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::create_images_from_range",
            "m_images is NULL");
        exit(EXIT_FAILURE);
    }

    size_t target_num_images = image_end - image_start;
    if (destination.cols() != target_num_images) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::create_images_from_range",
            "Destination Matrix size incorrect");
        if (MNIST_UTILS_DEBUG){ 
            Log::log_message(Log::Log_Priority::ERROR, "MNIST_Images::create_images_from_range",
                std::format("Destination Matrix size [{} x {}] but should be [{} x {}]",
                    destination.rows(), destination.cols(), MNIST_IMAGE_SIZE, target_num_images));
        }
        exit(EXIT_FAILURE);
    }

    // Iterate over the image Matrix instances and put into one giant Matrix
    for (size_t i = 0; i < target_num_images; ++i) {
        for (size_t j = 0; j < MNIST_IMAGE_HEIGHT; ++j) {
            for (size_t k = 0; k < MNIST_IMAGE_WIDTH; ++k) {
                destination.set((j * MNIST_IMAGE_WIDTH) + k, i, m_images[image_start + i]->get(j, k));
            }
        }
    }
}

bool MNIST_Labels::exists(size_t index) const {

    if (index >= m_num_labels) {
        Log::log_message(Log::Log_Priority::WARNING, "MNIST_Labels::exists",
            "Invalid index provided");
        if (MNIST_UTILS_DEBUG) {
            Log::log_message(Log::Log_Priority::DEBUG, "MNIST_Labels::exists",
                std::format("Requested index {} but only have {}", index, m_num_labels - 1));
        }
        return false;
    }
    if (m_labels == NULL) {
        Log::log_message(Log::Log_Priority::WARNING, "MNIST_Labels::exists",
            "m_labels is NULL. Not returning a value");
        return false;
    }

    return true;
}

MNIST_Labels::MNIST_Labels(const char* path) {

    // Open the path to where the labels are stored, in read-only mode
    FILE* labels_file = fopen(path, "ro");

    if (labels_file == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::MNIST_Labels",
            std::format("Unable to open path '{}' to MNIST labels. Exiting...",
                path));
        exit(EXIT_FAILURE);
    }

    // We want to read in two values at once here to avoid using fread twice
    uint32_t label_buffer[2] = {0, 0};

    // Read in 8 bytes from the file, grabbing the magic number and number of items contained
    if (fread(&label_buffer, sizeof(uint32_t), 2, labels_file) != 2) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::MNIST_Labels",
            "Unable to read headers from MNIST label file");
        fclose(labels_file);
        exit(EXIT_FAILURE);
    }

    // The first entry in the array is used for the magic number; the second is the number of items
    if (map_uint32(label_buffer[0]) != MNIST_LABEL_MAGIC) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::MNIST_Labels",
            "Mistmatched magic number in the MNIST label file header");
        fclose(labels_file);
        exit(EXIT_FAILURE);
    }

    // Store the number of labels we are expecting
    m_num_labels = (size_t)map_uint32(label_buffer[1]);
    if (MNIST_UTILS_DEBUG) {
        Log::log_message(Log::Log_Priority::INFO, "MNIST_Labels::MNIST_Labels",
            std::format("Reading {} labels", m_num_labels));
    }

    // Allocate memory for storing the labels themselves
    m_labels = (uint8_t*)calloc(m_num_labels, sizeof(uint8_t));
    if (m_labels == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::MNIST_Labels",
            "Unable to allocate memory to store labels");
        fclose(labels_file);
        exit(EXIT_FAILURE);
    }

    // Read in all the labels at once
    if (fread(m_labels, sizeof(uint8_t), m_num_labels, labels_file) != m_num_labels) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::MNIST_Labels",
            "Failed to read labels");
        fclose(labels_file);
        exit(EXIT_FAILURE);
    }

    // Cleanup
    fclose(labels_file);
}

MNIST_Labels::~MNIST_Labels() {

    if (m_labels != NULL) {
        free(m_labels);
        m_labels = NULL;
    }
    else {
        Log::log_message(Log::Log_Priority::WARNING, "MNIST_Labels::~MNIST_Labels",
            "m_labels already NULL. Skipping deallocation");
    }
}

size_t MNIST_Labels::size(void) const {
    return m_num_labels;
}

uint8_t MNIST_Labels::get(size_t index) const {

    if (!exists(index)) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::get",
            "Invalid index provided");
        if (MNIST_UTILS_DEBUG) {
            Log::log_message(Log::Log_Priority::DEBUG, "MNIST_Labels::get",
                std::format("Requested index {} but only have {}", index, m_num_labels - 1));
        }
        exit(EXIT_FAILURE);
    }
    return m_labels[index];
}

Matrix* MNIST_Labels::create_label(size_t index) const {

    if (!exists(index)) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::create_label",
            "Invalid index provided");
        exit(EXIT_FAILURE);
    }

    Matrix* result = new Matrix(MNIST_LABELS, 1);
    create_label(index, *result);
    return result;
}

void MNIST_Labels::create_label(size_t index, Matrix& destination) const {

    if (!exists(index)) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::create_label",
            "Invalid index provided");
        exit(EXIT_FAILURE);
    }

    // Zero out any values in the Matrix
    destination.populate(0);

    // Set only the row representing the value as a one, leaving the rest as zeros
    destination.set(m_labels[index], 0, 1);
}

Matrix* MNIST_Labels::create_labels_from_range(size_t label_start, size_t label_end) const {

    // Only check for exceeding the m_num_labels since size_t can't ever be negative
    if (label_start >= m_num_labels || label_end >= m_num_labels) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::create_labels_from_range",
            "label_start or label_end is out of range.");
        if (MNIST_UTILS_DEBUG) {
            Log::log_message(Log::Log_Priority::DEBUG, "MNIST_Labels::create_labels_from_range",
                std::format("Got range ({}, {}) but max index is {}",
                    label_start, label_end, m_num_labels - 1));
        }
        exit(EXIT_FAILURE);
    }
    if (m_labels == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::create_labels_from_range",
            "m_labels is NULL. Not returning a value");
        exit(EXIT_FAILURE);
    }

    Matrix* result = new Matrix(MNIST_LABELS, label_end - label_start);
    create_labels_from_range(label_start, label_end, *result);
    return result;
}

void MNIST_Labels::create_labels_from_range(size_t label_start, size_t label_end, Matrix& destination) const {

    // Only check for exceeding the m_num_labels since size_t can't ever be negative
    if (label_start >= m_num_labels || label_end >= m_num_labels) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::create_labels_from_range",
            "label_start or label_end is out of range.");
        if (MNIST_UTILS_DEBUG) {
            Log::log_message(Log::Log_Priority::DEBUG, "MNIST_Labels::create_labels_from_range",
                std::format("Got range ({}, {}) but max index is {}",
                    label_start, label_end, m_num_labels - 1));
        }
        exit(EXIT_FAILURE);
    }
    if (m_labels == NULL) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::create_labels_from_range",
            "m_labels is NULL. Not returning a value");
        exit(EXIT_FAILURE);
    }
    if ((label_end - label_start) != destination.cols()) {
        Log::log_message(Log::Log_Priority::ERROR, "MNIST_Labels::create_labels_from_range",
            "Number of labels to process and size of destination Matrix do not match");
        if (MNIST_UTILS_DEBUG) {
            Log::log_message(Log::Log_Priority::DEBUG, "MNIST_Labels::create_labels_from_range",
                std::format("Got number of labels {}, but Matrix has {} columns",
                    label_end - label_start, destination.cols()));
        }
        exit(EXIT_FAILURE);
    }
    // Clear out the contents of the destination Matrix
    destination.populate(0);

    // Iterate over the label range and set the value in each Matrix
    for (size_t i = 0; i < (label_end - label_start); ++i) {

        // Convert to size_t since we are going to use this as an index
        size_t label_value = (size_t)m_labels[label_start + i];

        // Set the row corresponding to the label_value as 1.0f
        destination.set(label_value, i, 1.0f);
    }
}

float MNIST_Utils_NS::pixel_to_float(const uint8_t* pixel) {

    return (float)*pixel / 255.0f;
}

uint32_t MNIST_Utils_NS::map_uint32(uint32_t in) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (
        ((in & 0xFF000000) >> 24) |
        ((in & 0x00FF0000) >>  8) |
        ((in & 0x0000FF00) <<  8) |
        ((in & 0x000000FF) << 24)
    );
#else
    return in;
#endif
}

