// == mojo ====================================================================
//
//    mnist_parser.h: prepares MNIST data for testing/training
//
//    This code was modified from tiny_cnn https://github.com/nyanp/tiny-cnn
//    It can parse MNIST data which you need to download and unzip locally on 
//    your machine. 
//    You can get it from: http://yann.lecun.com/exdb/mnist/index.html
//
// ==================================================================== mojo ==

/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#pragma once


/*#include <iostream> // cout
#include <sstream>
#include <fstream>
#include <iomanip> //setw*/
#include <random>
#include <stdio.h>


namespace mnist
{
std::string data_name() {return std::string("MNIST");}
//char *data_name() { char *ret = "MNIST";  return ret;}

// from tiny_cnn
template<typename T>
T* reverse_endian(T* p) {
	std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
	return p;
}

// from tiny_cnn (kinda)
//bool parse_mnist_labels(const std::string& label_file, std::vector<int> *labels) {
bool parse_mnist_labels(const char* label_file, std::vector<int> *labels) {
//	std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
    open_file(label_file);
   
    int magic_number, num_items;

	read_file((char*) &magic_number, 4);
	read_file((char*) &num_items, 4);
	
	reverse_endian(&magic_number);
	reverse_endian(&num_items);
	
	for (size_t i = 0; i < num_items; i++) {
		unsigned char label;
		read_file((char*) &label, 1);
		labels->push_back((int) label);
	}
	return true;
}

// from tiny_cnn
struct mnist_header {
	int magic_number;
	int num_items;
	int num_rows;
	int num_cols;
};

// from tiny_cnn (kinda)
bool parse_mnist_images(const char * image_file, 
	std::vector<std::vector<float>> *images,
	float scale_min = -1.0, float scale_max = 1.0,
	int x_padding = 0, int y_padding = 0) 
{
	open_file(image_file);
	
	mnist_header header;

	// read header
	read_file((char*) &header.magic_number, 4);
	read_file((char*) &header.num_items, 4);
	read_file((char*) &header.num_rows, 4);
	read_file((char*) &header.num_cols, 4);

	reverse_endian(&header.magic_number);
	reverse_endian(&header.num_items);
	reverse_endian(&header.num_rows);
	reverse_endian(&header.num_cols);

	const int width = header.num_cols + 2 * x_padding;
	const int height = header.num_rows + 2 * y_padding;
	
	// read each image
	for (size_t i = 0; i < header.num_items; i++) 
	{
		std::vector<float> image;
		std::vector<unsigned char> image_vec(header.num_rows * header.num_cols);

		read_file((char*) &image_vec[0], header.num_rows * header.num_cols);
		image.resize(width * height, scale_min);
	
		for (size_t y = 0; y < header.num_rows; y++)
		{
			for (size_t x = 0; x < header.num_cols; x++)
				image[width * (y + y_padding) + x + x_padding] = 
					(image_vec[y * header.num_cols + x] / 255.0f) * (scale_max - scale_min) + scale_min;
		}
		
		images->push_back(image);
	}

	return true;
}

// == load data (MNIST-28x28x1 size, no padding, pixel range -1 to 1)
bool parse_test_data(char *data_path, std::vector<std::vector<float>> &test_images, std::vector<int> &test_labels, 
	float min_val=-1.f, float max_val=1.f, int padx=0, int pady=0)
{
    char *d1, *d2, *d3, *d4; 
    d1 = strncat(d1, data_path, strlen(data_path)); d1 = strncat(d1, "/t10k-images.idx3-ubyte", strlen("/t10k-images.idx3-ubyte"));
    d2= strncat(d2, data_path, strlen(data_path)); d2 = strncat(d2, "/t10k-images-idx3-ubyte", strlen("/t10k-images-idx3-ubyte"));
    d3 = strncat(d3, data_path, strlen(data_path)); d3 = strncat(d1, "/t10k-labels.idx1-ubyte", strlen("/t10k-labels.idx1-ubyte"));
    d4 = strncat(d4, data_path, strlen(data_path)); d4 = strncat(d1, "/t10k-labels-idx1-ubyte", strlen("/t10k-labels-idx1-ubyte"));
    
	if(!parse_mnist_images(d1, &test_images, min_val, max_val, padx, pady)) 
		if (!parse_mnist_images(d2, &test_images, min_val, max_val, padx, pady))
			return false;
	if(!parse_mnist_labels(d3, &test_labels)) 
		if (!parse_mnist_labels(d4, &test_labels)) return false;
	return true;
}

bool parse_train_data(char *data_path, std::vector<std::vector<float>> &train_images, std::vector<int> &train_labels, 
	float min_val=-1.f, float max_val=1.f, int padx=0, int pady=0)
{
    char *d1, *d2, *d3, *d4; 
    d1 = strncat(d1, data_path, strlen(data_path)); d1 = strncat(d1, "/train-images.idx3-ubyte", strlen("/train-images.idx3-ubyte"));
    d2= strncat(d2, data_path, strlen(data_path)); d2 = strncat(d2, "/train-images-idx3-ubyte", strlen("/train-images-idx3-ubyte"));
    d3 = strncat(d3, data_path, strlen(data_path)); d3 = strncat(d1, "/train-labels.idx1-ubyte", strlen("/train-labels.idx1-ubyte"));
    d4 = strncat(d4, data_path, strlen(data_path)); d4 = strncat(d1, "/train-labels-idx1-ubyte", strlen("/train-labels-idx1-ubyte"));
    
	if(!parse_mnist_images(d1, &train_images, min_val, max_val, padx, pady))
		if (!parse_mnist_images(d2, &train_images, min_val, max_val, padx, pady))
			return false;
	if(!parse_mnist_labels(d3, &train_labels))
		if (!parse_mnist_labels(d4, &train_labels)) return false;
	return true;
}
}


