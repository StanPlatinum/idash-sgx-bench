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
	
void new_network(const char *model_file)
{
    cnn.enable_external_threads(); 
	cnn.set_learning_rate(initial_learning_rate);
	printf("model_file: %s\n", model_file);
    if(!cnn.read((char *)model_file)) 
    {
        printf("error: could not read model.\n");
    }
    
    printf(" Mojo CNN Configuration:\n");
	printf("%s\n\n", cnn.get_configuration().c_str());
}

int cnn_outsize()
{
    return cnn.out_size();
}

int classification(float *testimage, int size)
{  
    return cnn.predict_class(testimage);
}

float get_estimated_accuracy()
{
    return cnn.estimated_accuracy;
}
