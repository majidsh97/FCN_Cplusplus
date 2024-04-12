#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <concepts>

#include <fstream>
#include <sstream>
#include <cassert>

//random
#include <random>
#include <algorithm>
#include <tuple>
#include <math.h>
#include "Eigen/Dense"
#include <functional>

using std::cout;



template< typename ScalarType >
ScalarType stringToScalar(const std::string& str)
{
	std::stringstream s(str);
	ScalarType scalar;
	s >> scalar;
	return scalar;
}


template< class T >
concept Arithmetic = std::is_arithmetic_v< T >;
/*
template< Arithmetic ComponentType >
class Tensor ://public Eigen::Matrix<ComponentType, Eigen::Dynamic, Eigen::Dynamic>
	public Eigen::MatrixXd //Eigen::Matrix<ComponentType,-1,-1>
{
public:
	auto begin() { return this->data(); };
	auto end() { return this->data()+this->size() ; };
	//auto matmul(Tensor<ComponentType>& T1,){};
	ComponentType& operator[](size_t i) {

		return this->data()[i];
	};

	std::vector<Eigen::Index> shape() { return std::vector<Eigen::Index>({ this->rows(),this->cols()}); };
	//Tensor<ComponentType>& operator=( Eigen::MatrixXd& m) { this->data() =  m.data(); };



};
*/


template< Arithmetic ComponentType >
class Tensor
{
public:
	auto begin() { return m.data(); };
	auto end() { return m.data() + m.size(); };
	//auto matmul(Tensor<ComponentType>& T1,){};
	ComponentType& operator[](size_t i) {

		return m.data()[i];
	};

	std::vector<Eigen::Index> shape() { return std::vector<Eigen::Index>({ m.rows(),m.cols() }); };
	//Tensor<ComponentType>& operator=( Eigen::MatrixXd& m) { this->data() =  m.data(); };
	Eigen::MatrixXd m;


};

template<class T>
std::ostream& operator<<(std::ostream& o, std::vector<T>& v) {
	for (auto& i : v) {
		o << i << ' ';
	}
	return o;
}
template<class T>
std::ostream& operator<<(std::ostream& o, Tensor<T>& t) {
	o << std::endl;
	for (auto i = 0; i < t.m.rows(); i++) {
		o << i << " : ";
		for (int j = 0; j < t.m.cols(); j++) {
			o << t.m(i, j) << ' ';

		}
		o << '\n';

	}
	return o;
}
//---------------------------


int reverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
// Reads a tensor from file.


template< typename ComponentType >
Tensor< ComponentType > readMNISTTensorFromFile(const std::string& filename, int x = 32)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
	{
		std::cerr << "Could not open file." << std::endl;
		std::exit(1);
	}

	// MNIST header information
	int magic_number = 0;
	int number_of_items = 0;
	int rows = 0;
	int cols = 0;

	// Reading header
	file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
	magic_number = reverseInt(magic_number); // Adjust for endianness

	file.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items));
	number_of_items = reverseInt(number_of_items);

	file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
	rows = reverseInt(rows);

	file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
	cols = reverseInt(cols);
	//std::cout << number_of_items;
	// Prepare tensor
	x = static_cast<size_t>(x);
	rows = static_cast<size_t>(rows);
	cols = static_cast<size_t>(cols);
	Tensor<ComponentType> tensor({ x, rows * cols });

	// Read each image into the tensor
	for (size_t i = 0; i < x; ++i) {
		for (size_t r = 0; r < rows * cols; ++r) {

			unsigned char pixel;
			file.read(reinterpret_cast<char*>(&pixel), 1);
			tensor.m(i, r) = static_cast<ComponentType>(pixel) / 255;

		}
	}





	file.close();
	return tensor;
}
template< typename ComponentType >
Tensor< ComponentType > readMNISTLabelFromFile(const std::string& filename, int x = 32)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
	{
		std::cerr << "Could not open file." << std::endl;
		std::exit(1);
	}

	// MNIST header information
	int magic_number = 0;
	int number_of_items = 0;


	// Reading header
	file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
	magic_number = reverseInt(magic_number); // Adjust for endianness

	file.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items));
	number_of_items = reverseInt(number_of_items);

	// Prepare tensor
	x = static_cast<size_t>(x);
	Tensor<ComponentType> tensor({ x,1 });

	// Read each image into the tensor
	for (size_t i = 0; i < static_cast<size_t>(x); ++i) {

		unsigned char pixel;
		file.read(reinterpret_cast<char*>(&pixel), 1);

		tensor.m(i,0) = static_cast<ComponentType>(pixel);

	}

	file.close();
	return tensor;

}

void writeTensorToFile( const std::string& filename , std::function<void(std::ofstream&)> f)
{

	std::ofstream file;
	file.open(filename);
	f(file);
	/*
	file << tensor.rank() << "\n";
	for (auto d : tensor.shape())
	{
		file << d << "\n";
	}

	if (tensor.rank() == 0)
	{
		file << tensor({}) << "\n";
	}
	else
	{
		std::vector< size_t > idx(tensor.shape().size(), 0);
		size_t cnt = 0;
		while (cnt < tensor.numElements())
		{
			file << tensor(idx) << "\n";

			idx[tensor.rank() - 1]++;
			for (size_t i = tensor.rank() - 1; i > 0; i--)
			{
				if (idx[i] >= tensor.shape()[i])
				{
					idx[i] = 0;
					idx[i - 1]++;
				}
			}

			cnt++;
		}
	}
	*/
	file.close();
}