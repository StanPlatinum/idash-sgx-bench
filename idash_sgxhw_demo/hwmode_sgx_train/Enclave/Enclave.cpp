#include "Enclave_t.h"

#include <stdarg.h>  // va_list etc.
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// printf function is defined in mojo/util.h


#include "core_math.h"
#include "mojo.h"
#include "util.h"
#include "network.h"
//--------------- the above files passed

#include "solver.h" // only supported sgd, need more time here. should be fixed already?

#include "cost.h" 
#include "activation.h"
#include "layer.h"
#include "mnist_parser.h" // file IO for mnist images


void printf(const char *fmt, ...) { 
	char buf[BUFSIZ] = { '\0' };
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, BUFSIZ, fmt, ap);
	va_end(ap);
	ocall_print(buf);
}

int generate_random_number() {
//    printf("data_name: %s\n", data_name());
    ocall_print("Processing random number generation...");
    return 42;
}


//std::string model_file="../models/mnist_deepcnet.mojo";
//std::string model_file="../models/cifar_deepcnet.mojo";
std::string solver = "adam";
mojo::network cnn(solver.c_str());
const int mini_batch_size = 24;
const float initial_learning_rate = 0.0001f;  // This is important

/*
//	mojo::network cnn(solver.c_str());
	// !! the threading must be enabled with thread count prior to loading or creating a model !!
//	cnn.enable_external_threads();
//	cnn.set_mini_batch_size(mini_batch_size);
//	cnn.set_smart_training(true); // automate training
//	cnn.set_learning_rate(initial_learning_rate);
	
	// Note, network descriptions can be read from a text file with similar format to the API
//	cnn.read("../models/mnist_quickstart.txt");

	/*
	// to construct the model through API calls...
	cnn.push_back("I1", "input 28 28 1");				// MNIST is 28x28x1
	cnn.push_back("C1", "convolution 5 8 1 elu");		// 5x5 kernel, 20 maps. stride 1. out size is 28-5+1=24
	cnn.push_back("P1", "semi_stochastic_pool 3 3");	// pool 3x3 blocks. stride 3. outsize is 8
	cnn.push_back("C2i", "convolution 1 16 1 elu");		// 1x1 'inceptoin' layer
	cnn.push_back("C2", "convolution 5 48 1 elu");		// 5x5 kernel, 200 maps.  out size is 8-5+1=4
	cnn.push_back("P2", "semi_stochastic_pool 2 2");	// pool 2x2 blocks. stride 2. outsize is 2x2
	cnn.push_back("FC2", "softmax 10");					// 'flatten' of 2x2 input is inferred
	// connect all the layers. Call connect() manually for all layer connections if you need more exotic networks.
	cnn.connect_all();
	// */
	
void new_network(const char *model_file)
{
    cnn.enable_external_threads(); 
	cnn.set_mini_batch_size(mini_batch_size);
	cnn.set_smart_training(true); // automate training
	cnn.set_learning_rate(initial_learning_rate);
	    printf("model_file: %s\n", model_file);
    if(!cnn.read((char *)model_file)) 
    {
        printf("error: could not read model.\n");
    }
    
    printf(" Mojo CNN Configuration:\n");
	printf("%s\n\n", cnn.get_configuration().c_str());
	
	cnn.set_random_augmentation(2,2,0,0,mojo::edge); // padding
}

int cnn_outsize()
{
    return cnn.out_size();
}

int classification(float *testimage, int size)
{  
    return cnn.predict_class(testimage);
}

int get_epoch()
{
    return cnn.get_epoch();
}

void epoch(char *str)
{
    cnn.start_epoch(str);
}

void train(float *testimage, int size, int train_labels)
{
    cnn.train_class(testimage, train_labels);
}

void end_epoch()
{
    cnn.end_epoch();
}

float get_estimated_accuracy()
{
    return cnn.estimated_accuracy;
}

void reset_smart_training()
{
    cnn.reset_smart_training();
}

void write_model_file(char *model_file)
{
    cnn.write(model_file, true);
}

int elvis_left_the_building()
{
    return cnn.elvis_left_the_building();
}
