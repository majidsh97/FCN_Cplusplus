#pragma once
#include "tensor.hpp"
#include <math.h>
#include "BaseLayer.hpp"
#include "BaseLoss.hpp"

template < class C>
class SCCE : public BaseLoss<C> {
public:
	Tensor<C> forward(Tensor<C>& input_tensor, Tensor<C>& y_true);
	Tensor<C> backward(Tensor<C>& y_true);
private:
	Tensor<C> forward_value;
};


/*
template <class C>
Tensor<C> SCCE<C>::forward(Tensor<C>& input_tensor, Tensor<C>& y_true) {
	this->forward_value = input_tensor;
	auto end = input_tensor.shape()[0];
	//self.__prediction_tensor = prediction_tensor
	//loss = -np.sum(np.multiply(label_tensor, np.log(prediction_tensor + 2.22044604925e-16)))#, -1, keepdims = True) # 2.225e-16

	return Tensor<C>({ 1 },-loss);
};

template <class C>
Tensor<C> SCCE<C>::backward(Tensor<C>& y_true) {
	auto end = this->forward_value.shape()[0];
	Tensor<C> err(this->forward_value.shape(),0.0);
	for (size_t i = 0; i < end; i++) {
		err({i,static_cast<size_t>(y_true[i]) }) = -1 / (this->forward_value({i,static_cast<size_t>(y_true[i])}) + 2.22044604925e-16);
	}

	return err;
};
*/


template <class C>
Tensor<C> SCCE<C>::forward(Tensor<C>& input_tensor, Tensor<C>& y_true) {
	this->forward_value = input_tensor;
	auto end = input_tensor.shape()[0];
 	double loss = 0;
	//cout << y_true.shape();
	//cout << y_true;

	for (size_t i = 0; i < end; i++) {
		loss += std::log( input_tensor.m( i,(size_t)y_true.m(i,0) )  + 2.22044604925e-16);
	}
	
	return Tensor<C>(Eigen::MatrixXd(  { { - loss } } ) );
};

template <class C>
Tensor<C> SCCE<C>::backward(Tensor<C>& y_true) {
	auto shape = this->forward_value.shape();
	Tensor<C> err({shape[0] , shape[1]});
	err.m.setConstant(0.0);
	for (size_t i = 0; i < shape[0]; i++) {
		err.m(i,static_cast<size_t>(y_true.m(i,0)) ) = -1 / (this->forward_value.m(i,static_cast<size_t>(y_true.m(i,0) )) + 2.22044604925e-16);
	}

	return err;
};

