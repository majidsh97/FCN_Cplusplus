#pragma once
#include "tensor.hpp"
template<class C>
class BaseOptim {
public:
	BaseOptim() = default;
	virtual BaseOptim<C>* clone() { return new BaseOptim<C>(*this); };
	virtual Tensor<C>& update(Tensor<C>& weight, Tensor<C>& gradient);
	virtual ~BaseOptim() = default;
};

template<class C>
Tensor<C>& BaseOptim<C>::update(Tensor<C>& weight, Tensor<C>& gradient)
{
	// TODO: insert return statement here
	return weight;
}

template<class C>
class SGD : public BaseOptim<C> {
public:
	SGD(double lr);
	Tensor<C>& update(Tensor<C>& weight, Tensor<C>& gradient) override;
	void set_lr(double lr) { this->lr = lr; };
	double get_lr() { return this->lr; };

	SGD<C>* clone() { return new SGD<C>(*this); };
private:
	double lr;

};
template<class C>
SGD<C>::SGD(double lr) :lr(lr) {

}

template<class C>
inline Tensor<C>& SGD<C>::update(Tensor<C>& weight, Tensor<C>& gradient)
{
	weight.m = weight.m - lr * gradient.m;
	return weight;
}


template<class C>
class SGDWM :public BaseOptim<C>
{
public:
	SGDWM(double lr = 0.001, double mom = 0.9) :lr(lr), momentum_rate(mom) {
//		v.m = Eigen::MatrixXd::Zero(1, 1);
	};
	Tensor<C>& update(Tensor<C>& weight, Tensor<C>& gradient) {

		//		self.v = self.momentum_rate*self.v - self.learning_rate * gradient_tensor
		if (v.m.rows() == 0 && v.m.cols()==0) {
			v.m = -lr * gradient.m;
		}else{
			v.m = momentum_rate * v.m - lr * gradient.m;

		}
		//		new_weight = weight_tensor + self.v + self.calculate_gradient(weight_tensor)
		weight.m = weight.m + v.m;
		return weight;

	}
	SGDWM<C>* clone() { return new SGDWM<C>(*this); };

private:
	double lr;
	double momentum_rate;
	Tensor<C> v;

};