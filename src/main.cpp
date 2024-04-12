#include "tensor.hpp"
#include "Relu.hpp"
#include "Linear.hpp"
#include "optim.hpp"
#include "SCCE.hpp"
#include "SoftMax.hpp"
#include "NN.hpp"
using namespace std;

//#include <omp.h>


/*
#include <Dense>
#include "src/tensor.hpp"
#include "Linear.hpp"
#include "Init.hpp"
#include "Loss.hpp"
#include "NN.hpp"
#include "optim.hpp"
#include "SoftMax.hpp"
#include "SCCE.hpp"
*/
//#include <torch/torch.h>
//#include "tests/tests.hpp"

// only works with release
// add path to additional include c/c++ project property and add libs path to linker proj property
//add c++ version in general proj property for both debug and release
//#include "matplotlibcpp.h"
//using std::cout;
//class MyTest;

//TODO: multi thread, memory return?, matrix mult, bias , SoftMax, loss BCe

//namespace plt = matplotlibcpp;
/*
class MyTest {
public:
	MyTest();
	void test_linear();
	void data();

private:
	Tensor<double> x;
	Tensor<double> y;
	//torch::Tensor tx;
	//torch::Tensor ty;
	Linear<double> l1;
	Linear<double> l2;
	std::vector< BaseLayer<double>* > layers;
	MSE<double> mse;
	SGD<double> sgd;;
	NN<double> nn;


};

MyTest::MyTest() : x({ 20, 1 }), y({ 20, 1 }), tx(x), ty(y), l1(1, 50), l2(50, 1),sgd(0.001) {
	data();
	tx = x;
	ty = y;

	layers
		= {
		&l1,
		&l2
	};



	// Create the neural network
	nn = NN<double>(layers, &mse);
	nn.compile(sgd);
}

void MyTest::test_linear() {

	auto o = nn.forward(x);
	cout << o;
}
void MyTest::data() {

	Init<double>::uniform_inplace(y);
	for (size_t i = 0; i < 20; ++i)
	{
		x({ i,0 }) = i;
		y({ i,0 }) += i;
	}
}
*/


#include <iostream>
#include <fstream>
#include <string>
#include <map>

std::map<std::string, std::string> readVariablesFromFile(const std::string& filename) {
	// Define the variable names to search for
	std::string variables[] = {
		"rel_path_train_images",
		"rel_path_train_labels",
		"rel_path_test_images",
		"rel_path_test_labels",
		"rel_path_log_file",
		"num_epochs",
		"batch_size",
		"hidden_size",
		"learning_rate"
	};

	// Initialize a map to store variable-value pairs
	std::map<std::string, std::string> variable_map;

	// Open the text file
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening file\n";
		return variable_map; // Return an empty map on error
	}

	// Read the file line by line
	std::string line;
	while (std::getline(file, line)) {
		// Search for variable names
		for (const auto& variable : variables) {
			size_t found = line.find(variable);
			if (found != std::string::npos) {
				// Extract the value by splitting the line
				size_t equal_sign_pos = line.find('=');
				if (equal_sign_pos != std::string::npos) {
					size_t comment_pos = line.find("//");
					std::string value = line.substr(equal_sign_pos + 1, comment_pos - equal_sign_pos - 1);
					// Trim leading and trailing whitespace
					value.erase(0, value.find_first_not_of(" \t\r\n"));
					value.erase(value.find_last_not_of(" \t\r\n") + 1);
					// Add the variable-value pair to the map
					variable_map[variable] = value;
				}
			}
		}
	}

	// Close the file
	file.close();

	return variable_map;
}




void MyParallelFunction(int id) {
	std::cout << "Thread " << id << ": Executing parallel function" << std::endl;
};

void read_image(string image_dataset_input, string image_tensor_output, int image_index) {
	//cout << image_dataset_input << image_index;
	Tensor<double> x = readMNISTTensorFromFile< double >(image_dataset_input, image_index + 1);
	auto y = x.m.row(image_index).array();
	//cout << y;
	writeTensorToFile(image_tensor_output, [&y, image_index](ofstream& file) {
		file << 2 << endl;
		file << 28 << endl;
		file << 28 << endl;


		for (auto& i : y) {
			file << i << endl;
		}

		});
	//
};
void read_label(string image_dataset_input, string image_tensor_output, int image_index) {
	//cout << image_dataset_input << image_index << endl;
	//cout<<"label start";
	Tensor<double> x = readMNISTLabelFromFile< double >(image_dataset_input, image_index + 1);
	int y = x.m(image_index);
	//cout << "label :";
	Eigen::MatrixXd onehot = Eigen::MatrixXd::Zero(10, 1);
	onehot(y, 0) = 1.0;
	//cout << onehot;
	//cout << endl;
	writeTensorToFile(image_tensor_output, [&onehot, &image_index](ofstream& file) {
		file << 1 << endl;
		file << 10 << endl;
		file << onehot;

		});
		cout<<"label done";

};
int main(int argc, char** argv)
{

	string what;
	if (argc > 1) {
		what = (string)argv[1];
	}
	string image_dataset_input = "";
	string image_tensor_output = "";
	int image_index = 0;

	//cout << what;
	if (what == "image") {
		image_dataset_input = (string)argv[2];
		image_tensor_output = (string)argv[3];
		image_index = stoi(argv[4]);
		read_image(image_dataset_input, image_tensor_output, image_index);
	}
	else
		if (what == "label") {

			//read_label();
			image_dataset_input = (string)argv[2];
			image_tensor_output = (string)argv[3];
			image_index = stoi(argv[4]);
			read_label(image_dataset_input, image_tensor_output, image_index);
		}
		else
			if (what == "mnist") {
				string config_filename = (string)argv[2];


				std::map<std::string, std::string> config = readVariablesFromFile(config_filename);

				// Print the resulting map


				/*		"rel_path_train_images",
					"rel_path_train_labels",
					"rel_path_test_images",
					"rel_path_test_labels",
					"rel_path_log_file",
					"num_epochs",
					"batch_size",
					"hidden_size",
					"learning_rate"*/

					//------------------------------------------------------------------------------------

				int batch_size = stoi(config["batch_size"]);
				int num_epochs = stoi(config["num_epochs"]);
				double learning_rate = stod(config["learning_rate"]);
				auto X = readMNISTTensorFromFile< double >(config["rel_path_train_images"], batch_size);
				auto Y = readMNISTLabelFromFile<double>(config["rel_path_train_labels"], batch_size);

				auto X_Test = readMNISTTensorFromFile< double >(config["rel_path_test_images"], batch_size);
				auto Y_Test = readMNISTLabelFromFile<double>(config["rel_path_test_labels"], batch_size);


				int kk = stoi(config["hidden_size"]);
				auto l1 = Linear<double>(X.m.cols(), kk);

				auto r1 = Relu<double>();

				//auto l2 = Linear<double>(kk, kk);
				//auto r2 = Relu<double>();
				auto l3 = Linear<double>(kk, 10);


				auto softmax = SoftMax<double>();

				std::vector< BaseLayer<double>* > layers
					= {
					&l1,
					&r1,
					//&l2,
					//&r2,


					&l3,
					&softmax

				};
				SCCE<double> scce;
				NN<double> nn(layers, &scce);

				//cout << Y;
				auto x1 = nn.forward(X);
				auto acc = nn.acc(x1, Y);
				//cout << "\nAcc: " << acc << " %\n";
				//cout << X;
				//cout << x1;
				SGDWM<double> sgd(learning_rate);
				nn.compile(sgd);



				for (int i = 0; i < num_epochs; i++) {
					//auto o = nn.forward(x);
					auto o = nn.train(X, Y);
					if (i % 1 == 0) {
						cout << i << ": " << o[0] << std::endl;
						auto x1 = nn.forward(X);
						auto acc = nn.acc(x1, Y);
						nn.logger(x1, Y, /*batch????*/0, config["rel_path_log_file"]);

					}
					//if (i == 100)
						//sgd.set_lr( lr/10);
					//cout << l1.get_weights() << l1.get_bias();
				}

				//-----------------------------------------------------------------------------------
				//cout << "\nAcc: "<< acc<<" %\n";


			   //auto y = l1.forward(X);
			   //auto yy = l1.backward(y);
			   //cout << y;
			}
	int x;
	cin >> x;

	return 0;

}


