#pragma once
#include "tensor.hpp"
#include "BaseLayer.hpp"

template<class C>
class Relu : public BaseLayer<C> {
public:
	Tensor<C>& forward(Tensor<C>& input_tensor);
	Tensor<C>& backward(Tensor<C>& error_tensor);

};

template<class C>
Tensor<C>& Relu<C>::forward(Tensor<C>& input_tensor) {

	input_tensor.m = input_tensor.m.cwiseMax(0.0);
	this->forward_value = input_tensor;
	return input_tensor;
}

template<class C>
Tensor<C>& Relu<C>::backward(Tensor<C>& error_tensor) {
	Eigen::MatrixXd x = this->forward_value.m.unaryExpr([](const C& val) { return val > 0 ? 1.0 : 0.0; });
	//(this->forward_value.m.array() > 0).cast<double>();
	error_tensor.m = error_tensor.m.cwiseProduct(x);

	return error_tensor;
}