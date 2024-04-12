#pragma once
#include "tensor.hpp"
#include "optim.hpp"
template<class C>
class BaseLayer {
public:
	BaseLayer();
	virtual Tensor<C>& forward(Tensor<C>& input_tensor);
	virtual Tensor<C>& backward(Tensor<C>& error_tensor);
	void setOptim(BaseOptim<C>* opt, BaseOptim<C>* opt_bias);
	virtual ~BaseLayer() {
		if (opt != nullptr) {
			delete opt;
		}
		if (opt_bias != nullptr) {
			delete opt_bias;
		}
	};

protected:
	Tensor<C> forward_value;
	bool trainable;
	BaseOptim<C>* opt = nullptr;
	BaseOptim<C>* opt_bias = nullptr;



};

template<class C>
BaseLayer<C>::BaseLayer() :forward_value(), trainable(false), opt(nullptr), opt_bias(nullptr) {

}

template<class C>
Tensor<C>& BaseLayer<C>::forward(Tensor<C>& input_tensor)
{
	// TODO: insert return statement here
	return input_tensor;
}

template<class C>
Tensor<C>& BaseLayer<C>::backward(Tensor<C>& error_tensor)
{
	// TODO: insert return statement here
	return error_tensor;
}

template<class C>
void BaseLayer<C>::setOptim(BaseOptim<C>* opt, BaseOptim<C>* opt_bias)
{
	if (this->opt != nullptr) {
		delete this->opt;
	}
	if (this->opt_bias != nullptr) {
		delete this->opt_bias;
	}
	this->opt = opt;
	this->opt_bias = opt_bias;

}
