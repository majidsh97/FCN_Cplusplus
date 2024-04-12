#pragma once
#include "tensor.hpp"
#include "BaseLayer.hpp"
#include "optim.hpp"

template< class C>
class Linear : public BaseLayer<C> {
public:
	Linear(size_t input_shape, size_t output_shape);
	Tensor<C>& forward(Tensor<C>& input_tensor) override;
	Tensor<C>& backward(Tensor<C>& error_tensor) override;
	Tensor<C>& get_weights();
	Tensor<C>& get_bias();

private:
	Tensor<C> weights;
	Tensor<C> bias;


};

template<class C>
Linear<C>::Linear(size_t input_shape, size_t output_shape) :weights({ input_shape,output_shape }), bias({ 1,output_shape }) {
	this->trainable = true;
	//Init<C>::normal_inplace(weights,0, 1.0);
	this->weights.m.setRandom();
	weights.m = weights.m * 1/ sqrt(2*(input_shape + output_shape));
	//Init<C>::uniform_inplace(weights,-1.0/ input_shape,1.0/ input_shape);

	this->bias.m.setConstant(0.0);
	//Init<C>::normal_inplace(bias);
};

template<class C>
Tensor<C>& Linear<C>::forward(Tensor<C>& input_tensor) {
	// x * w
	this->forward_value = input_tensor;
	 input_tensor.m *= weights.m;

	auto b = input_tensor.shape()[0];
	auto os = input_tensor.shape()[1];
	for (size_t i = 0; i < b; i++) {
		for (size_t j = 0; j < os; j++) {
			input_tensor.m( i,j ) += this->bias.m( 0,j );
		}
	}

	return input_tensor;
}

template<class C>
Tensor<C>& Linear<C>::backward(Tensor<C>& error_tensor) {
	auto fvt = this->forward_value.m.transpose();
	auto bias_grad = Tensor<C>( error_tensor.m.colwise().sum()); //error_tensor.sum(1);

	//auto gradient_weights = Tensor<C>( (fvt * error_tensor).eval()) ; // err*inp.t
	auto gradient_weights = Tensor<C>( fvt * error_tensor.m); // err*inp.t


	auto wt = this->weights.m.transpose();
	error_tensor.m *=  wt;
	
	// err * w.t
	this->weights = this->opt->update(this->weights, gradient_weights);
	this->bias = this->opt_bias->update(this->bias, bias_grad);
	

	return error_tensor;
}

template<class C>
Tensor<C>& Linear<C>::get_weights()
{
	// TODO: insert return statement here
	return this->weights;
}

template<class C>
Tensor<C>& Linear<C>::get_bias()
{
	// TODO: insert return statement here
	return this->bias;
}
