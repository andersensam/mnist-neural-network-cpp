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
 * @version: 2025-07-19
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

/* Control debug settings */
#define MATRIX_DEBUG 1

/* Standard dependencies */
#include <iostream>
#include <string.h>
#include <math.h>

/* Local dependencies */
#include "Log.hpp"

/* Definitions */

namespace Matrix_NS{

typedef enum {
    ROW,
    COLUMN
} Vector_Orientation;

typedef enum {
    ADD,
    SUBTRACT,
    MULTIPLY
} Element_Operations;

template <typename Matrix_Type> class Matrix {
private:
    /* Private data elements */
    Matrix_Type* m_data = NULL;
    size_t m_num_rows = 0;
    size_t m_num_cols = 0;

    /* Helper functions that aren't ever used publicly */

    /**
     * Generate a string for an error message detailing the size
     * mismatch between two Matrix instances
     * @param target The Matrix to compare dimensions with
     * @returns Returns a formatted string
     */
    const std::string dimension_mismatch(const Matrix<Matrix_Type>& target) const {
        return std::format("Dimension mismatch. Calling Matrix has dimensions [{} x {}] and other Matrix has [{} x {}]", 
            rows(), cols(), target.rows(), target.cols());
    }

    /**
     * Generate a string for an error message detailing the size
     * mismatch between the underlying data pointerrs
     * @param target The Matrix to comapre size with
     * @returns Returns a formatted string
     */
    const std::string size_mismatch(const Matrix<Matrix_Type>& target) const {
        return std::format("Size mismatch. Calling Matrix has underlying data size {} and the other Matrix has size {}",
            size(), target.size());
    }

    /**
     * Perform a generic operation on all elements of a source and
     * a target Matrix, storing in a specified destination. This function should
     * only be called after validating sizes match properly
     * @param target Matrix to perform the operation with
     * @param destination Destination Matrix for the result
     * @param operation The operation to perform
     */
    void element_op(const Matrix<Matrix_Type>& target, Matrix<Matrix_Type>& destination,
        Element_Operations operation) const {

        if (operation == Element_Operations::ADD) {
            for (size_t i = 0; i < rows() * cols(); ++i) {
                destination.m_data[i] = m_data[i] + target.m_data[i];
            }
        }
        else if (operation == Element_Operations::SUBTRACT) {
            for (size_t i = 0; i < rows() * cols(); ++i) {
                destination.m_data[i] = m_data[i] - target.m_data[i];
            }
        }
        else if (operation == Element_Operations::MULTIPLY) {
            for (size_t i = 0; i < rows() * cols(); ++i) {
                destination.m_data[i] = m_data[i] * target.m_data[i];
            }
        }
        else {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::element_op",
                "Invalid element operation provided");
        }
     }

     /**
     * Perform a generic operation on all elements of a source and
     * a target Matrix, overwriting the calling Matrix. This function should
     * only be called after validating sizes match properly
     * @param target Matrix to perform the operation with
     * @param operation The operation to perform
     */
    void element_op(const Matrix<Matrix_Type>& target, Element_Operations operation) {

        if (operation == Element_Operations::ADD) {
            for (size_t i = 0; i < rows() * cols(); ++i) {
                m_data[i] += target.m_data[i];
            }
        }
        else if (operation == Element_Operations::SUBTRACT) {
            for (size_t i = 0; i < rows() * cols(); ++i) {
                m_data[i] -= target.m_data[i];
            }
        }
        else if (operation == Element_Operations::MULTIPLY) {
            for (size_t i = 0; i < rows() * cols(); ++i) {
                m_data[i] *= target.m_data[i];
            }
        }
        else {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::element_op",
                "Invalid element operation provided");
        }
     }

    /**
     * Perform a generic operation on all elements of a Matrix, overwriting the calling
     * Matrix instance
     * @param value The value to be applied
     * @param operation The operation to perform
     */
    void element_op(Matrix_Type value, Element_Operations operation) {

        if (operation == Element_Operations::ADD) {
            for (size_t i = 0; i < rows() * cols(); ++i) {
                m_data[i] += value;
            }
        }
        else if (operation == Element_Operations::SUBTRACT) {
            for (size_t i = 0; i < rows() * cols(); ++i) {
                m_data[i] -= value;
            }
        }
        else if (operation == Element_Operations::MULTIPLY) {
            for (size_t i = 0; i < rows() * cols(); ++i) {
                m_data[i] *= value;
            }
        }
        else {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::element_op",
                "Invalid element operation provided");
        }
    }

    /**
     * Apply a function to a Matrix, overwriting the caller
     * @param func Function to apply, this function must return a value of type Matrix_Type
     */
    void apply_fn(Matrix_Type (*func)(Matrix_Type)) {

        for (size_t i = 0; i < rows() * cols(); ++i) {
            m_data[i] = (*func)(m_data[i]);
        }
    }

    /**
     * Apply a function to a MAtrix, overwriting the caller
     * @param func Function to apply, this function must return a value of type Matrix_Type
     * @param param Additional parameter to be passed to func, must be of type Matrix_Type
     */
    void apply_fn(Matrix_Type (*func)(Matrix_Type, Matrix_Type), Matrix_Type param) {

        for (size_t i = 0; i < rows() * cols (); ++i) {
            m_data[i] = (*func)(m_data[i], param);
        }
    }

    /**
     * Validate the dimensions of two Matrix instances
     * @param target Target Matrix to validate against
     * @param caller Function this is being called from
     * @returns True if they match, False otherwise
     */
    bool correct_dimensions(const Matrix<Matrix_Type>& target, const char* caller) const {

        if (rows() != target.rows() || cols() != target.cols()) {
            Log::log_message(Log::Log_Priority::ERROR, caller,
                "Dimension mismatch");
            
            if (MATRIX_DEBUG) {
                Log::log_message(Log::Log_Priority::DEBUG, caller,
                    dimension_mismatch(target));
            }
            return false;
        }
        return true;
    }

    /**
     * Validate the size of the underlying data matches between two Matrix instances
     * @param target Target Matrix to validate against
     * @param caller Calling function
     * @returns Returns True if they match, false otherwise
     */
    bool correct_sizes(const Matrix<Matrix_Type>& target, const char* caller) const {

        if (size() != target.size()) {
            Log::log_message(Log::Log_Priority::ERROR, caller,
                "Underlying data sizes do not match");

            if (MATRIX_DEBUG) {
                Log::log_message(Log::Log_Priority::DEBUG, caller,
                    size_mismatch(target));
            }
            return false;
        }
        return true;
    }

public:
    /* Public functions */
    
    /**
     * Create a new Matrix of type Matrix_Type
     * @param num_rows Number of rows
     * @param num_cols Number of colums
     * @returns Returns a new Matrix
     */
    Matrix(size_t num_rows, size_t num_cols) {

        m_num_rows = num_rows;
        m_num_cols = num_cols;
        /* Allocate memory for data storage */
        m_data = (Matrix_Type*)calloc(num_rows * num_cols, sizeof(Matrix_Type));
        /* Zero out the memory manually, to be safe */
        memset(m_data, '\0', num_cols * num_rows * sizeof(Matrix_Type));
    }

    /**
     * Destructor for Matrix
     */
    ~Matrix() {
        if (m_data != NULL) { 
            free(m_data);
            m_data = NULL;
        }
    }

    /**
     * Get the number of rows present in a Matrix
     * @returns Returns the number of rows
     */
    size_t rows(void) const {
        return m_num_rows;
    }

    /**
     * Get the number of columns present in a Matrix
     * @returns Returns the number of columns
     */
    size_t cols(void) const {
        return m_num_cols;
    }

    /**
     * Get the size of the underlying data for a Matrix
     * @returns Returns the m_data size
     */
    size_t size(void) const {
        return sizeof(Matrix_Type) * rows() * cols();
    }

    /**
     * Check whether the index provided is valid for the Matrix
     * @param target_row Row to check
     * @param target_col Column to check
     * @returns True if valid, False if invalid
     */
    bool exists(size_t target_row, size_t target_col) const {

        if (target_row <= m_num_rows - 1 
            && target_col <= m_num_cols - 1 
            && m_data != NULL) { 
                return true; 
        }

        Log::log_message(Log::Log_Priority::ERROR, "Matrix::exists", 
            "Invalid row or column provided");

        if (MATRIX_DEBUG) {
            Log::log_message(Log::Log_Priority::DEBUG, "Matrix::exists",
                std::format("Dimensions [{} x {}]", m_num_rows, m_num_cols));
        }

        return false;
    }

    /**
     * Get the value at a coordinate within a Matrix
     * @param target_row Row to fetch from
     * @param target_col Column to fetch from
     * @returns Returns the value at (target_row, target_col)
     */
    Matrix_Type get(size_t target_row, size_t target_col) const {

        if (exists(target_row, target_col)) {
            return m_data[(target_row * this->m_num_cols) + target_col];
        }

        Log::log_message(Log::Log_Priority::ERROR, "Matrix::get",
            "Invalid coordinate provided to get. Exiting now to prevent a crash");
        exit(EXIT_FAILURE);
    }

    /**
     * Set the value at a coordinate within a Matrix
     * @param target_row Row to set
     * @param target_col Column to seet
     * @param data Data to write
     */
    void set(size_t target_row, size_t target_col, Matrix_Type data) {

        if (exists(target_row, target_col)) {
            m_data[(target_row * this->m_num_cols) + target_col] = data;
            return;
        }

        Log::log_message(Log::Log_Priority::ERROR, "Matrix::set",
            "Invalid coordinate provided to set. Exiting now to prevent a crash");
        exit(EXIT_FAILURE);
    }

    /**
     * Print a Matrix
     */
    void print(void) const {

        for (size_t i = 0; i < rows(); ++i) {
            std::cout << std::format("[{}]:\t", i);
            for (size_t j = 0; j < cols(); ++j) {
                std::cout << get(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

    /**
     * Get the maximum value stored in a Matrix
     * @returns Returns the max value in the Matrix
     */
    Matrix_Type max(void) const {

        Matrix_Type current_max = m_data[0];

        for (size_t i = 1; i < m_num_rows * m_num_cols; ++i) {

            Matrix_Type current_value = m_data[i];
            current_max = (current_max > current_value) ? current_max : current_value;
        }
        return current_max;
    }

    /**
     * Get the maximum value stored in a vector (slice of a Matrix)
     * @param orientation Either ROW or COLUMN
     * @param index The ROW or COLUMN index to search through
     * @returns Returns the max of the vector
     */
    Matrix_Type max(Vector_Orientation orientation, size_t index) const {

        Matrix_Type current_max = m_data[0];

        if (orientation == Vector_Orientation::ROW) {
            if (!exists(index, 0)) {

                Log::log_message(Log::Log_Priority::ERROR, "Matrix::max",
                    "Invalid row provided to max");
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < m_num_cols; ++i) {

                Matrix_Type current_value = get(index, i);
                current_max = (current_max > current_value) ? current_max : current_value;
            }
        }
        else {
            if (!exists(0, index)) {

                Log::log_message(Log::Log_Priority::ERROR, "Matrix::max",
                    "Invalid column provided to max");
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < m_num_rows; ++i) {

                Matrix_Type current_value = get(i, index);
                current_max = (current_max > current_value) ? current_max : current_value;
            }
        }
        return current_max;
    }

    /**
     * Get the minimum value stored in a Matrix
     * @returns Returns the minimum value in the Matrix
     */
    Matrix_Type min(void) const {

        Matrix_Type current_min = m_data[0];

        for (size_t i = 1; i < m_num_rows * m_num_cols; ++i) {

            Matrix_Type current_value = m_data[i];
            current_min = (current_min < current_value) ? current_min : current_value;
        }
        return current_min;
    }

    /**
     * Get the minimum value stored in a vector (slice of a Matrix)
     * @param orientation Either ROW or COLUMN
     * @param index The ROW or COLUMN index to search through
     * @returns Returns the min of the vector
     */
    Matrix_Type min(Vector_Orientation orientation, size_t index) const {

        Matrix_Type current_min = m_data[0];

        if (orientation == Vector_Orientation::ROW) {
            if (!exists(index, 0)) {
                Log::log_message(Log::Log_Priority::ERROR, "Matrix::min",
                    "Invalid row provided to min");
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < m_num_cols; ++i) {

                Matrix_Type current_value = get(index, i);
                current_min = (current_min < current_value) ? current_min : current_value;
            }
        }
        else {
            if (!exists(0, index)) {
                Log::log_message(Log::Log_Priority::ERROR, "Matrix::min",
                    "Invalid row provided to min");
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < m_num_rows; ++i) {

                Matrix_Type current_value = get(i, index);
                current_min = (current_min < current_value) ? current_min : current_value;
            }
        }
        return current_min;
    }
    
    /**
     * Compute the dot product of two Matrix instances
     * @param target Matrix to calculate the dot product with
     * @returns Returns a new Matrix instance with the result
     */
    Matrix<Matrix_Type>* dot(const Matrix<Matrix_Type>& target) const {

        if (cols() != target.rows()) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::dot",
                "Matrix dimension mismatch. Cannot calculate the dot product");

            if (MATRIX_DEBUG) {
                Log::log_message(Log::Log_Priority::DEBUG, "Matrix::dot",
                    std::format("First Matrix is [{} x {}], second is [{} x {}]",
                        rows(), cols(), target.rows(), target.cols()));
            }
            exit(EXIT_FAILURE);
        }

        Matrix<Matrix_Type>* result = new Matrix<Matrix_Type>(rows(), target.cols());
        /* Use the other dot method since it implements the same logic */
        dot(target, *result);
        return result;
    }

    /**
     * Compute the dot product of two Matrix instances, saving the result
     * to an existing Matrix
     * @param target Matrix to calculate the dot product with
     * @param destination Destination Matrix
     */
    void dot(const Matrix<Matrix_Type>& target, Matrix<Matrix_Type>& destination) const {

        if (cols() != target.rows()) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::dot",
                "Matrix dimension mismatch. Cannot calculate the dot product");

            if (MATRIX_DEBUG) {
                Log::log_message(Log::Log_Priority::DEBUG, "Matrix::dot",
                    std::format("First Matrix is [{} x {}], second is [{} x {}]",
                        rows(), cols(), target.rows(), target.cols()));
            }
            exit(EXIT_FAILURE);
        }

        if (destination.rows() != rows() || destination.cols() != target.cols()) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::dot",
                "Destination Matrix has the wrong dimensions");

            if (MATRIX_DEBUG) {
                Log::log_message(Log::Log_Priority::DEBUG, "Matrix::dot",
                    std::format("Destination Matrix is [{} x {}], but should be [{} x {}]",
                        destination.rows(), destination.cols(), rows(), target.cols()));
            }
            exit(EXIT_FAILURE);
        }

        Matrix_Type sum = 0;

        /* Iterate over the expected rows of the new Matrix */
        for (size_t i = 0; i < destination.rows(); ++i) {
            for (size_t j = 0; j < destination.cols(); ++j) {
                /* Sum the multiplications of the elements from self + target */
                for (size_t k = 0; k < cols(); ++k) {
                    sum += get(i, k) * target.get(k, j);
                }
                destination.set(i, j, sum);
                sum = 0;
            }
        }
    }

    /**
     * Flatten a Matrix to either one row or column, depending on the desired orientation
     * @param orientation Either ROW or COLUMN
     */
    void flatten(Vector_Orientation orientation) {

        if (orientation == Vector_Orientation::ROW) {
            m_num_cols = m_num_rows * m_num_cols;
            m_num_rows = 1;
        }
        else {
            m_num_rows = m_num_rows * m_num_cols;
            m_num_cols = 1;
        }
    }

    /**
     * Clones a Matrix, doing a deep copy of the data
     * @returns Returns a new Matrix instance
     */
    Matrix<Matrix_Type>* clone(void) const {

        Matrix<Matrix_Type>* target = new Matrix<Matrix_Type>(rows(), cols());
        memcpy(target->m_data, m_data, size());
        return target;
    }

    /**
     * Copies the contents of one Matrix to another
     * @param destination Destination to write to
     */
    void copy_to(Matrix<Matrix_Type>& destination) const {

        if (!correct_sizes(destination, "Matrix::copy_to")) {
            exit(EXIT_FAILURE);
        }

        memcpy(destination.m_data, m_data, size());
        destination.m_num_rows = rows();
        destination.m_num_cols = cols();
    }

    /**
     * Populate a Matrix with a value
     * @param value Value to populate the Matrix with
     */
    void populate(Matrix_Type value) {

        for (size_t i = 0; i < rows() * cols(); ++i) {
            m_data[i] = value;
        }
    }

    /**
     * Transpose a Matrix
     * @returns Returns a pointer to the transpose of the Matrix
     */
    Matrix<Matrix_Type>* transpose(void) const {

        Matrix<Matrix_Type>* result = new Matrix<Matrix_Type>(cols(), rows());

        for (size_t i = 0; i < m_num_rows; ++i) {
            for (size_t j = 0; j < m_num_cols; ++j) {
                result->set(j, i, get(i, j));
            }
        }
        return result;
    }

    /**
     * Transpose a Matrix, overwriting itself
     */
    void transpose_self(void) {

        //Matrix<Matrix_Type>* t = transpose();
        //t->copy_to(*this);
        //delete t;

        Matrix<Matrix_Type> copy_of_self = Matrix<Matrix_Type>(rows(), cols());
        copy_to(copy_of_self);

        // Swap the rows and columns
        m_num_rows = copy_of_self.cols();
        m_num_cols = copy_of_self.rows();

        for (size_t i = 0; i < copy_of_self.rows(); ++i) {
            for (size_t j = 0; j < copy_of_self.cols(); ++j) {
                set(j, i, copy_of_self.get(i, j));
            }
        }
    }

    /**
     * Add two Matrix instances together
     * @param target Matrix to add with
     * @returns Returns a pointer to a new Matrix with the addition
     */
    Matrix<Matrix_Type>* add(const Matrix<Matrix_Type>& target) const {

        if (!correct_dimensions(target, "Matrix::add")) {
            exit(EXIT_FAILURE);
        }
        Matrix<Matrix_Type>* result = new Matrix<Matrix_Type>(rows(), cols());
        add(target, *result);
        return result;
    }

    /**
     * Add two Matrix instances together, storing in a specified destination
     * @param target Matrix to add
     * @param destination Destination Matrix to store the result
     */
    void add(const Matrix<Matrix_Type>& target, Matrix<Matrix_Type>& destination) const {

        if (!correct_dimensions(target, "Matrix::add")) {
            exit(EXIT_FAILURE);
        }
        if (!correct_sizes(destination, "Matrix::add")) {
            exit(EXIT_FAILURE);
        }
        destination.m_num_rows = rows();
        destination.m_num_cols = cols();
        element_op(target, destination, Element_Operations::ADD);
    }

    /**
     * Add two Matrix instances together, overwriting the calling Matrix
     * @param target Matrix to add
     */
    void add_o(const Matrix<Matrix_Type>& target) {

        if (!correct_dimensions(target, "Matrix::add_o")) {
            exit(EXIT_FAILURE);
        }
        element_op(target, Element_Operations::ADD);
    }

     /**
     * Subtract two Matrix instances together
     * @param target Matrix to subtract with
     * @returns Returns a pointer to a new Matrix with the subtraction
     */
    Matrix<Matrix_Type>* subtract(const Matrix<Matrix_Type>& target) const {

        if (!correct_dimensions(target, "Matrix::subtract")) {
            exit(EXIT_FAILURE);
        }
        Matrix<Matrix_Type>* result = new Matrix<Matrix_Type>(rows(), cols());
        subtract(target, *result);
        return result;
    }

    /**
     * Subtract two Matrix instances together, storing in a specified destination
     * @param target Matrix to subtract
     * @param destination Destination Matrix to store the result
     */
    void subtract(const Matrix<Matrix_Type>& target, Matrix<Matrix_Type>& destination) const {

        if (!correct_dimensions(target, "Matrix::subtract")) {
            exit(EXIT_FAILURE);
        }
        if (!correct_sizes(destination, "Matrix::subtract")) {
            exit(EXIT_FAILURE);
        }
        destination.m_num_rows = rows();
        destination.m_num_cols = cols();
        element_op(target, destination, Element_Operations::SUBTRACT);
    }

    /**
     * Subtract two Matrix instances together, overwriting the calling Matrix
     * @param target Matrix to subtract
     */
    void subtract_o(const Matrix<Matrix_Type>& target) {

        if (!correct_dimensions(target, "Matrix::subtract_o")) {
            exit(EXIT_FAILURE);
        }
        element_op(target, Element_Operations::SUBTRACT);
    }

    /**
     * Multiply two Matrix instances together
     * @param target Matrix to multiply with
     * @returns Returns a pointer to a new Matrix with the multiplication
     */
    Matrix<Matrix_Type>* multiply(const Matrix<Matrix_Type>& target) const {

        if (!correct_dimensions(target, "Matrix::multiply")) {
            exit(EXIT_FAILURE);
        }
        Matrix<Matrix_Type>* result = new Matrix<Matrix_Type>(rows(), cols());
        multiply(target, *result);
        return result;
    }

    /**
     * Subtract two Matrix instances together, storing in a specified destination
     * @param target Matrix to multiply
     * @param destination Destination Matrix to store the result
     */
    void multiply(const Matrix<Matrix_Type>& target, Matrix<Matrix_Type>& destination) const {

        if (!correct_dimensions(target, "Matrix::multiply")) {
            exit(EXIT_FAILURE);
        }
        if (!correct_sizes(destination, "Matrix::multiply")) {
            exit(EXIT_FAILURE);
        }
        destination.m_num_rows = rows();
        destination.m_num_cols = cols();
        element_op(target, destination, Element_Operations::MULTIPLY);
    }

    /**
     * Multiply two Matrix instances together, overwriting the calling Matrix
     * @param target Matrix to multiply with
     */
    void multiply_o(const Matrix<Matrix_Type>& target) {

        if (!correct_dimensions(target, "Matrix::multiply_o")) {
            exit(EXIT_FAILURE);
        }
        element_op(target, Element_Operations::MULTIPLY);
    }

    /**
     * Scale a Matrix, i.e. multiply by a scalar value elementwise, storing the 
     * result in a new Matrix instance
     * @param value Scalar value to multiply with
     * @returns Returns a new Matrix instance with the scalar applied
     */
    Matrix<Matrix_Type>* scale(Matrix_Type value) const {

        Matrix<Matrix_Type>* result = clone();
        result->element_op(value, Element_Operations::MULTIPLY);
        return result;
    }

    /**
     * Scale a Matrix, i.e. multiply by a scalar value elementwise, storing
     * the result in a predefined destination
     * @param value Scalar value to multiply with
     * @param destination Matrix to store the value in
     */
    void scale(Matrix_Type value, Matrix<Matrix_Type>& destination) const {

        if (!correct_sizes(destination, "Matrix::scale")) {
            exit(EXIT_FAILURE);
        }
        copy_to(destination);
        destination.element_op(value, Element_Operations::MULTIPLY);
    }

    /**
     * Scale a Matrix, i.e. multiply by a scalar value elementwise,
     * overwriting the calling Matrix
     * @param value Scalar value to multiply with
     */
    void scale_o(Matrix_Type value) {

        element_op(value, Element_Operations::MULTIPLY);
    }

    /**
     * Add a scalar value to a Matrix, elementwise, storing the result
     * in a new Matrix instance
     * @param value Value to add
     */
    Matrix<Matrix_Type>* add_scalar(Matrix_Type value) const {

        Matrix<Matrix_Type>* result = clone();
        result->element_op(value, Element_Operations::ADD);
        return result;
    }

    /**
     * Add a scalar value to a Matrix, elementwise, storing the result
     * in a predefined destination
     * @param value Value to add
     * @param destination Matrix to store the value in
     */
    void add_scalar(Matrix_Type value, Matrix<Matrix_Type>& destination) const {

        if (!correct_sizes(destination, "Matrix::add_scalar")) {
            exit(EXIT_FAILURE);
        }
        copy_to(destination);
        destination.element_op(value, Element_Operations::ADD);
    }

    /**
     * Add a scalar value to a Matrix, elementwise, overwriting the calling Matrix
     * @param value Value to add
     */
    void add_scalar_o(Matrix_Type value) {

        element_op(value, Element_Operations::ADD);
    }

    /**
     * Apply a function to a Matrix, storing the result in a new Matrix instance
     * @param func A function pointer that we want to use. The function must return
     * a value of Matrix_Type
     * @returns Returns a new Matrix instance with the function applied
     */
    Matrix<Matrix_Type>* apply(Matrix_Type (*func)(Matrix_Type)) const {

        Matrix<Matrix_Type>* result = clone();
        result->apply_fn(func);
        return result;
    }

    /**
     * Apply a function to a Matrix, storing the result in an existing Matrix
     * @param func A function pointer that we want to use. The function must return
     * a value of Matrix_Type
     * @param destination The destination Matrix
     */
    void apply(Matrix_Type (*func)(Matrix_Type), Matrix<Matrix_Type>& destination) const {

        if (!correct_sizes(destination, "Matrix::apply")) {
            exit(EXIT_FAILURE);
        }
        copy_to(destination);
        destination.apply_fn(func);
    }

    /**
     * Apply a function to a Matrix, overwriting the calling Matrix
     * @param func A function pointer that we want to use. The function must return
     * a value of Matrix_Type
     */
    void apply_o(Matrix_Type (*func)(Matrix_Type)) {

        apply_fn(func);
    }

    /**
     * Applies a function to a Matrix, taking in two parameters instead of one
     * @param func Function pointer that returns a value of type Matrix_Type
     * @param param Additional parameter to be passed to func
     * @returns Returns a new Matrix instance with the function applied
     */
    Matrix<Matrix_Type>* apply_second(Matrix_Type (*func)(Matrix_Type, Matrix_Type), Matrix_Type param) const {

        Matrix<Matrix_Type>* result = clone();
        result->apply_fn(func, param);
        return result;
    }

    /**
     * Apply a function to a Matrix, storing the result in an existing Matrix
     * The function must accept two parameters of type Matrix_Type
     * @param function Function pointer that returns a value of type Matrix_Type
     * @param param Additional parameter to be passed to func
     * @param destination The destination Matrix
     */
    void apply_second(Matrix_Type (*func)(Matrix_Type, Matrix_Type), Matrix_Type param, Matrix<Matrix_Type>& destination) const {

        if (!correct_sizes(destination, "Matrix::apply_second")) {
            exit(EXIT_FAILURE);
        }
        copy_to(destination);
        destination.apply_fn(func, param);
    }

    /**
     * Apply a function to a Matrix, overwriting the calling Matrix. This variant
     * takes two parameters to the function
     * @param func Function pointer that returns a value of type Matrix_Type
     * @param param Additional parameter to be passed to func
     */
    void apply_second_o(Matrix_Type (*func)(Matrix_Type, Matrix_Type), Matrix_Type param) {

        apply_fn(func, param);
    }

    /**
     * Apply a function to a row of a Matrix, overwriting the calling Matrix
     * @param func Function pointer that returns a value of type Matrix_Type
     * @param index Index of the row to apply the function to
     */
    void apply_row_o(Matrix_Type (*func)(Matrix_Type), size_t index) {

        if (!exists(index, 0)) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::apply_row_o",
                "Invalid index provided");
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < cols(); ++i) {
            set(index, i, func(get(index, i)));
        }
    }

    /**
     * Apply a function to a row of a Matrix, overwriting the calling Matrix
     * @param func Function pointer that returns a value of type Matrix_Type
     * @param param Additional parameter that func accepts
     * @param index Index of the row to apply the function to
     */
    void apply_second_row_o(Matrix_Type (*func)(Matrix_Type, Matrix_Type), Matrix_Type param, size_t index) {

        if (!exists(index, 0)) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::apply_second_row_o",
                "Invalid index provided");
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < cols(); ++i) {
            set(index, i, func(get(index, i), param));
        }
    }

    /**
     * Apply a function to a column of a Matrix, overwriting the calling Matrix
     * @param func Function pointer that returns a value of type Matrix_Type
     * @param index Index of the column to apply the function to
     */
    void apply_column_o(Matrix_Type (*func)(Matrix_Type), size_t index) {

        if (!exists(index, 0)) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::apply_column_o",
                "Invalid index provided");
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < rows(); ++i) {
            set(i, index, func(get(i, index)));
        }
    }

    /**
     * Apply a function to a column of a Matrix, overwriting the calling Matrix
     * @param func Function pointer that returns a value of type Matrix_Type
     * @param param Additional parameter that func accepts
     * @param index Index of the column to apply the function to
     */
    void apply_second_column_o(Matrix_Type (*func)(Matrix_Type, Matrix_Type), Matrix_Type param, size_t index) {

        if (!exists(index, 0)) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::apply_second_column_o",
                "Invalid index provided");
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < rows(); ++i) {
            set(i, index, func(get(i, index), param));
        }
    }

    /**
     * Get an entire row from a Matrix, storing the row in a new Matrix instance
     * @param row Row number to extract from the Matrix
     * @returns Returns a new Matrix of 1 row x N columns
     */
    Matrix<Matrix_Type>* get_row(size_t row) const {

        if (!exists(row, 0)) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::get_row",
                "Invalid row provided to get_row. Exiting");

            exit(EXIT_FAILURE);
        }

        Matrix<Matrix_Type>* result = new Matrix<Matrix_Type>(1, cols());
        
        for (size_t i = 0; i < cols(); ++i) {

            result->set(0, i, get(row, i));
        }

        return result;
    }

    /**
     * Get an entire row from a Matrix, storing the row in a predefined destination Matrix
     * @param row Row number to extrat from the Matrix
     * @param destination Reference to a Matrix to write to
     */
    void get_row(size_t row, Matrix<Matrix_Type>& destination) const {

        if (!exists(row, 0)) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::get_row",
                "Invalid row provided to get_row. Exiting");

            exit(EXIT_FAILURE);
        }

        if (destination.size() != sizeof(Matrix_Type) * cols()) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::get_row",
                "Destination Matrix does not have the correct m_data size");

            exit(EXIT_FAILURE);
        }

        if (destination.rows() != rows() || destination.cols() != cols()) {
            destination.m_num_rows = 1;
            destination.m_num_cols = cols();
        }

        for (size_t i = 0; i < cols(); ++i) {

            destination.set(0, i, get(row, i));
        }
    }

    /**
     * Get an entire column from a Matrix, storing the column in a new Matrix instance
     * @param col Column number to extract from the Matrix
     * @returns Returns a new Matrix of N rows x 1 column
     */
    Matrix<Matrix_Type>* get_column(size_t col) const {

        if (!exists(0, col)) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::get_col",
                "Invalid column provided to get_col. Exiting");

            exit(EXIT_FAILURE);
        }

        Matrix<Matrix_Type>* result = new Matrix<Matrix_Type>(rows(), 1);
        
        for (size_t i = 0; i < rows(); ++i) {

            result->set(i, 0, get(i, col));
        }

        return result;
    }

    /**
     * Get an entire column from a Matrix, storing the column in a predefined destination
     * @param col Column number to extract from the Matrix
     * @param destination Destination Matrix
     */
    void get_column(size_t col, Matrix<Matrix_Type>& destination) const {

        if (!exists(0, col)) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::get_column",
                "Invalid column provided to get_column. Exiting");

            exit(EXIT_FAILURE);
        }

        if (destination.size() != sizeof(Matrix_Type) * rows()) {
            Log::log_message(Log::Log_Priority::ERROR, "Matrix::get_column",
                "Destination Matrix does not have the correct m_data size");

            exit(EXIT_FAILURE);
        }

        if (destination.rows() != rows() || destination.cols() != cols()) {
            destination.m_num_rows = rows();
            destination.m_num_cols = 1;
        }

        for (size_t i = 0; i < rows(); ++i) {

            destination.set(i, 0, get(i, col));
        }
    }

    /**
     * Sum all the values in a Matrix
     * @returns Returns the sum of all values in the Matrix
     */
    Matrix_Type sum(void) const {

        Matrix_Type running_sum = 0;

        for (size_t i = 0; i < rows() * cols(); ++i) {
            running_sum += m_data[i];
        }

        return running_sum;
    }

    /**
     * Get the index of the maximum value in a vector (1D Matrix)
     * @param orientation Orientation we want to process the search in
     * @param index The row or column to search in
     * @returns Returns a size_t representing the index of the max value
     */
    size_t max_idx(Vector_Orientation orientation, size_t index) const {

        size_t max_index = 0;
        Matrix_Type max_value = 0;

        // Handle the case where the search doesn't require extracting
        // any one row or column
        if (rows() == 1 || cols() == 1) {

            max_value = max();

            if (orientation == Vector_Orientation::ROW) {
                for (size_t i = 0; i < cols(); ++i) {
                    if (get(0, i) == max_value) { return i; }
                }
            }
            else {
                for (size_t i = 0; i < rows(); ++i) {
                    if (get(i, 0) == max_value) { return i; }
                }
            }
        }

        // Handle the complicated case where we need to get the vector
        // before trying to find the max index
        if (orientation == Vector_Orientation::ROW) {
            Matrix<Matrix_Type> search = Matrix<Matrix_Type>(1, cols());
            get_row(index, search);
            max_index = search.max_idx(orientation, 0);
        }
        else {
            Matrix<Matrix_Type> search = Matrix<Matrix_Type>(rows(), 1);
            get_column(index, search);
            max_index = search.max_idx(orientation, 0);
        }

        return max_index;
    }

    /**
     * Get the sum of the absolute values of each element inside a Matrix
     * @returns Returns the absolute value of the sum
     */
    Matrix_Type abs_sum(void) const {

        Matrix_Type total_sum = 0;

        for (size_t i = 0; i < rows() * cols(); ++i) {
            total_sum += abs(m_data[i]);
        }

        return total_sum;
    }

};

};

#endif
