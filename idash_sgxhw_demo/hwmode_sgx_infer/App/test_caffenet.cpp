// == mojo ====================================================================
//
//    Copyright (c) gnawice@gnawice.com. All rights reserved.
//	  See LICENSE in root folder
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files(the "Software"),
//    to deal in the Software without restriction, including without 
//    limitation the rights to use, copy, modify, merge, publish, distribute,
//    sublicense, and/or sell copies of the Software, and to permit persons to
//    whom the Software is furnished to do so, subject to the following 
//    conditions :
//
//    The above copyright notice and this permission notice shall be included
//    in all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
//    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
//    OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
//    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ============================================================================
//    test.cpp:  Simple example using pre-trained model to test mojo cnn
//
//    Instructions: 
//	  Add the "mojo" folder in your include path.
//    Download MNIST data and unzip locally on your machine:
//		(http://yann.lecun.com/exdb/mnist/index.html)
//    Download CIFAR-10 data and unzip locally on your machine:
//		(http://www.cs.toronto.edu/~kriz/cifar.html)
//    Set the data_path variable in the code to point to your data location.
// ==================================================================== mojo ==

#include "Enclave_u.h"
#include "sgx_urts.h"

#include <iostream> // cout
#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include <mojo.h>

#include <unistd.h>

/* modified by Weijie */
#include "parser/imagenet_parser.h"

using namespace imagenet;

std::string data_path="../data/";
//std::string model_file="../models/mnist_deepcnet.mojo";
std::string model_file="../models/caffenet.txt"; //../models/snapshots/

/*/
#include "cifar_parser.h"
using namespace cifar;
std::string data_path="../data/cifar-10-batches-bin/";
std::string model_file="../models/cifar_deepcnet.mojo";
//*/

/* Global EID shared by multiple threads */
sgx_enclave_id_t eid = 0;

// OCall implementations
void ocall_print(const char* str) {
    printf("%s\n", str);
}

void mojo_sleep(unsigned milliseconds) 
{ 
    mojo::usleep(milliseconds * 1000); // ?
} 

FILE *f = NULL;
void open_file(const char* str)
{
    f = fopen(str, "r");
    if(!f)
    {
        printf("open image file error.\n");
        exit(0);
    }
}

void read_file(char *dest, int sz)
{
    fread(dest, sz, 1, f);
}


void close_file()
{
    if(f) fclose(f);
}

// deep network file operations follow
FILE *fnetwork = NULL;
FILE *output_network_file = NULL;

int open_networkfile(const char* str)
{
    fnetwork = fopen(str, "rb");
    if(!fnetwork)
    {
        printf("open deep network file error.\n");
        return 1;
    }
    
    return 0;
}

int open_outputnetworkfile(const char* str)
{
    output_network_file = fopen(str, "wb");
    if(!output_network_file)
    {
        printf("open deep network file error.\n");
        return 1;
    }
    
    return 0;
}

// OCall implementations, for open_outputnetworkfile
void ocall_fprint_networkfile(const char* str) {
    fprintf(output_network_file, "%s", str);
}

void ocall_write(char *src, int sz)
{
    fwrite(src, 1, sz, fnetwork);
}

////////////////

// OCall implementations
char ocall_fread_networkfile() {
    char ret = fgetc(fnetwork);
}

int ocall_getint()
{
    int ret;
    fscanf(fnetwork, "%d", &ret);
    
    return ret;
}

float ocall_getfloat()
{
    float ret;
    fscanf(fnetwork, "%f", &ret);
    
    return ret;
}

void ocall_read(char *src, int sz)
{
    fread(src, 1, sz, fnetwork);
}

void ocall_read_outenclave(uint64_t src, int sz)
{
//    printf("address: %p, size: %d\n", src, sz);

//	mojo::matrix *m = (mojo::matrix *)src;
			
    fread((char *)src, 1, sz, fnetwork);
    
//    printf("loaded.\n");
}

void end_this_line()
{
    char s[256];

    fgets(s, 256, fnetwork);
}

void close_networkfile()
{
    fflush(fnetwork);
    if(fnetwork) fclose(fnetwork);
}

void close_outputnetworkfile()
{
    fflush(output_network_file);
    if(output_network_file) fclose(output_network_file);
}

uint64_t ocall_newmatrix(uint64_t * px,  int *size, int cols, int rows, int chans)
{
    mojo::matrix *ret = new mojo::matrix(cols, rows, chans);
    //printf("OCALL: %p %p, size: %d, %d, cols, rows = %d %d\n", ret, ret->x, ret->size(), ret->_size, cols, rows);
    *px = (uint64_t) ret->x; 
    *size = ret->size();
    
    return (uint64_t)ret;
}

void ocall_fill_uniform(uint64_t pmatrix, float range)
{
    mojo::matrix *m = (mojo::matrix *)pmatrix;
    m->fill_random_uniform(range);
}

void ocall_fill_normal(uint64_t pmatrix, float std)
{
    mojo::matrix *m = (mojo::matrix *)pmatrix;
    m->fill_random_normal(std);
}

void test(const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels)
{
	int out_size;
	cnn_outsize(eid, &out_size); // we know this to be 10 for MNIST and CIFAR
	int correct_predictions=0;

	// use progress object for simple timing and status updating
	mojo::progress progress((int)test_images.size(), "  testing : ");

	const int record_cnt= (int)test_images.size();

	// when MOJO_OMP is defined, we use standard "omp parallel for" loop, 
	// the number of threads determined by network.enable_external_threads() call
	#pragma omp parallel for reduction(+:correct_predictions) schedule(dynamic)  // dynamic schedule just helps the progress class to work correcly
	for(int k=0; k<record_cnt; k++) 
	{
		// predict_class returnes the output index of the highest response
	//	printf("=================== record %d ==========\n", k);
		int prediction = 0;
		
	//	printf("test image size: %d\n", test_images[k].size());
		classification(eid, &prediction, (float *)test_images[k].data(), test_images[k].size()); // input data
   
//   std::cout<<"prediction: "<<prediction<<std::endl;
	//	const int prediction=cnn.predict_class(test_images[k].data());
		if(prediction ==test_labels[k]) correct_predictions++;
		if(k%100==0) progress.draw_progress(k);
	}
	float dt = progress.elapsed_seconds();
	std::cout << "  test time: " << dt << " seconds                                          "<< std::endl;
	std::cout << "  records: " << test_images.size() << std::endl;
	std::cout << "  speed: " << (float)record_cnt/dt << " records/second" << std::endl;
	std::cout << "  accuracy: " << (float)correct_predictions/record_cnt*100.f <<"%" << std::endl;
}


int main()
{
    sgx_status_t        ret = SGX_SUCCESS;
	sgx_launch_token_t  token = { 0 };
	int updated = 0;

	ret = sgx_create_enclave("enclave.signed.so", SGX_DEBUG_FLAG, &token, &updated, &eid, NULL);
	if (ret != SGX_SUCCESS)
		return -1;
	// Initializing the enclave finished.	
	
	int randnum;
	
	// == parse data
	// array to hold image data (note that mojo does not require use of std::vector)
	std::vector<std::vector<float>> test_images;
	// array to hold image labels 
	std::vector<int> test_labels;
	// calls MNIST::parse_test_data  or  CIFAR10::parse_test_data depending on 'using'
	if(!parse_test_data(data_path, test_images, test_labels)) {std::cerr << "error: could not parse data.\n"; return 1;}

	// == setup the network  
//	mojo::network cnn;
    
    new_network(eid, model_file.c_str());
	// here we need to prepare mojo cnn to store data from multiple threads
	// !! enable_external_threads must be set prior to loading or creating a model !!
//	cnn.enable_external_threads(); 

	// load model
//	if(!cnn.read(model_file)) {std::cerr << "error: could not read model.\n"; return 1;}
//	std::cout << "Mojo CNN Configuration:" << std::endl;
//	std::cout << cnn.get_configuration() << std::endl << std::endl;

	// == run the test
	std::cout << "Testing " << data_name() << ":" << std::endl;
	// this function will loop through all images, call predict, and print out stats
	test(test_images, test_labels);	

	std::cout << std::endl;
	
	sgx_destroy_enclave(eid);
	return 0;
}
