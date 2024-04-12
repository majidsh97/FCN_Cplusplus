#pragma once
#include "tensor.hpp"
#include "BaseLoss.hpp"

template < class C>
class MSE : public BaseLoss<C> {

public:

	Tensor<C> forward(Tensor<C>& input_tensor, Tensor<C>& y_true) override;
	//return error_tensor
	Tensor<C> backward(Tensor<C>& input_tensor, Tensor<C>& y_true) override;
};

template<class C>
Tensor<C> MSE<C>::forward(Tensor<C>& input_tensor, Tensor<C>& y_true)
{
	auto i = input_tensor.mul(-1);
	auto l = y_true.matsum(y_true, i);
	l = l.power2();
	l = l.sum(0);
	l = l.sum(1);
	return l;
	//C c;
	//return c;
}

template<class C>
Tensor<C> MSE<C>::backward(Tensor<C>& input_tensor, Tensor<C>& y_true)
{
	// TODO: insert return statement here
	auto i = input_tensor.mul(-1);
	auto a = y_true.matsum(y_true, i);
	a = a.mul(2);
	return a;

}

