#pragma once
#include "tensor.hpp"
#include <math.h>
#include "BaseLayer.hpp"
#include "Eigen/Dense"
template<class C>
class SoftMax : public BaseLayer<C> {
public:
	Tensor<C>& forward(Tensor<C>&);
	Tensor<C>& backward(Tensor<C>&);
};

template<class C>
Tensor<C>& SoftMax<C>::forward(Tensor<C>& input_tensor) {
	
	input_tensor.m =  (input_tensor.m.array() - input_tensor.m.maxCoeff()).exp() ;

	// Calculate sum along the last dimension (axis -1)
	auto s = input_tensor.m.rowwise().sum();

	// Calculate softmax output
	input_tensor.m = input_tensor.m.array().colwise() / s.array();
	this->forward_value = input_tensor;
	return input_tensor;

};


template<class C>
Tensor<C>& SoftMax<C>::backward(Tensor<C>& error_tensor) {
	//auto x = error_tensor.m.dot(this->forward_value.m);
	//auto x = (error_tensor.m.dot( this->forward_value.m)).colwise().sum();
	//auto x =  (error_tensor.m.array().rowwise() - ();
	//error_tensor.m= this->forward_value.m*
	

	auto y = (error_tensor.m.cwiseProduct(this->forward_value.m)).rowwise().sum();
	error_tensor.m = this->forward_value.m.cwiseProduct( error_tensor.m.colwise() - y );
	//error_tensor.m = b;

			//softmax_grad(i, j) = this->forward_value.m(i, j) * ();


	return error_tensor;

	//fuck
}



/*

#include <iostream>
#include <vector>

std::vector<std::vector<double>> softmax_backward(const std::vector<std::vector<double>>& grad_output, const std::vector<std::vector<double>>& output) {
	int batch_size = grad_output.size();
	int num_classes = grad_output[0].size();

	std::vector<std::vector<double>> grad_input(batch_size, std::vector<double>(num_classes, 0.0));

	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < num_classes; ++j) {
			double sum = 0.0;
			for (int k = 0; k < num_classes; ++k) {
				sum += output[i][k] * (k == j ? 1.0 - output[i][j] : -output[i][j]);
			}
			grad_input[i][j] = grad_output[i][j] * sum;
		}
	}

	return grad_input;
}

int main() {
	// Example usage
	std::vector<std::vector<double>> grad_output = {{0.1, 0.2, 0.3}, {0.2, 0.3, 0.5}};
	std::vector<std::vector<double>> output = {{0.3, 0.4, 0.3}, {0.2, 0.3, 0.5}};

	std::vector<std::vector<double>> grad_input = softmax_backward(grad_output, output);

	// Print the result
	for (const auto& row : grad_input) {
		for (double val : row) {
			std::cout << val << " ";
		}
		std::cout << std::endl;
	}

	return 0;
}



*/