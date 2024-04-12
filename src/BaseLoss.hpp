
#pragma once
#include "tensor.hpp"
template <class C>
class BaseLoss
{
public:
	//BaseLoss() = default;
	virtual Tensor<C> forward(Tensor<C>& input_tensor, Tensor<C>& y_true)=0;
	virtual Tensor<C> backward( Tensor<C>& y_true)=0;
	virtual ~BaseLoss() = default;
};

/*
template<class C>
Tensor<C> BaseLoss<C>::forward(Tensor<C>& input_tensor, Tensor<C>& y_true)
{

	return input_tensor;
}

template<class C>
inline Tensor<C> BaseLoss<C>::backward(Tensor<C>& input_tensor, Tensor<C>& y_true)
{
	return input_tensor;
}

*/