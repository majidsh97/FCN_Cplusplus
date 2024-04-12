#pragma once
#include "Linear.hpp"
//#include "Relu.hpp"
#include "optim.hpp"
#include "tensor.hpp"
#include <vector>
#include "BaseLayer.hpp"
#include "Loss.hpp"
using namespace std;
template <class C>
class NN {

public:
	NN(std::vector<BaseLayer<C>*>& layers, BaseLoss<C>* loss);
	NN() = default;
	~NN() = default;
	Tensor<C> train(Tensor<C>& input_tensor, Tensor<C>& y_true);
	Tensor<C> test(Tensor<C>& input_tensor, Tensor<C>& y_true);
	Tensor<C> forward(Tensor<C>& input_tensor);
	void backward(Tensor<C>& error_tensor);
	void compile(BaseOptim<C>&);
	double acc(Tensor<C>& input_tensor, Tensor<C>& y_true);
	void logger(Tensor<C>& input_tensor, Tensor<C>& y_true,int batch_num, std::string rel_path_log_file="log_predictions.txt" );
	std::vector<BaseLayer<C>*> layers;
	BaseLoss<C>* loss;
};

template<class C>
NN<C>::NN(std::vector<BaseLayer<C>*>& layers, BaseLoss<C>* loss) :layers(layers), loss(loss) {};

template<class C>
void NN<C>::logger(Tensor<C>& input_tensor, Tensor<C>& y_true, int batch_num, std::string rel_path_log_file){


	writeTensorToFile(rel_path_log_file, [&](ofstream& file) {
		file << "Current batch: " << batch_num << endl;
		Eigen::Index index;
		for (size_t i = 0; i < input_tensor.m.rows(); i++) {
			input_tensor.m.row(i).maxCoeff(&index);
			file << "- image " << i << ": Prediction=" << index << ". Label=" << y_true.m(i, 0) << endl;

		}


		});
		
};
template<class C>
double NN<C>::acc(Tensor<C>& input_tensor, Tensor<C>& y_true) {

	Eigen::Index index;
	size_t n = 0;
	for (size_t i = 0; i < input_tensor.m.rows(); i++) {
		input_tensor.m.row(i).maxCoeff(&index);
		if (y_true.m(i, 0) == index) {
			n++;
		}

	}
	double accs = n / (double)input_tensor.m.rows();
	cout << "Acc: " << accs * 100 << " %\n";
	return accs;
};


template<class C>
Tensor<C> NN<C>::forward(Tensor<C>& input_tensor) {
	auto o = input_tensor;
	for (auto& l : this->layers) {
		o = l->forward(o);
	}
	return o;
}

template<class C>
void NN<C>::backward(Tensor<C>& error_tensor) {
	auto o = error_tensor;
	for (auto l = this->layers.rbegin(); l != this->layers.rend(); ++l) {

		o = (*l)->backward(o);
	}
}



template<class C>
Tensor<C> NN<C>::train(Tensor<C>& input_tensor, Tensor<C>& y_true) {
	/*
	return loss
	*/
	auto o = this->forward(input_tensor);
	auto loss_value = this->loss->forward(o, y_true);
	auto error_tensor = this->loss->backward(y_true);
	this->backward(error_tensor);
	return loss_value;
}

template<class C>
Tensor<C> NN<C>::test(Tensor<C>& input_tensor, Tensor<C>& y_true) {
	auto o = this->forward(input_tensor);
	auto loss_value = this->loss->forward(o, y_true);
	return loss_value;
}

//TODO:: bias_optim need to be defined
/*template<class C>
void NN<C>::compile(BaseOptim<C>& opt) {
	for (auto& l : layers) {
		l->setOptim(&opt, &opt);
	}
}
*/

template<class C >
void NN<C>::compile(BaseOptim<C>& obj) {
	for (auto& l : layers) {
		l->setOptim(obj.clone(), obj.clone());
	}
}