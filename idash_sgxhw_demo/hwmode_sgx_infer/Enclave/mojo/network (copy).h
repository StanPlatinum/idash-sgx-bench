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
//    network.h: The main artificial neural network graph for mojo
// ==================================================================== mojo ==

#pragma once

#include <string>
#include <iostream> // cout
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

#include "layer.h"
#include "solver.h"
#include "activation.h"
#include "cost.h"

// hack for VS2010 to handle c++11 for(:)
#if (_MSC_VER  == 1600)
	#ifndef __for__
	#define __for__ for each
	#define __in__ in
	#endif
#else
	#ifndef __for__
	#define __for__ for
	#define __in__ :
	#endif
#endif



#if defined(MOJO_CV2) || defined(MOJO_CV3)

#ifdef MOJO_CV2
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#pragma comment(lib, "opencv_core249")
#pragma comment(lib, "opencv_highgui249")
#pragma comment(lib, "opencv_imgproc249")
#pragma comment(lib, "opencv_contrib249")
#else  //#ifdef MOJO_CV3
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#pragma comment(lib, "opencv_world310")
#endif
#endif


#define blocksize 100000

namespace mojo {

#if defined(MOJO_CV2) || defined(MOJO_CV3)
// forward declare these for data augmentation
cv::Mat matrix2cv(const mojo::matrix &m, bool uc8 = false);
mojo::matrix cv2matrix(cv::Mat &m);
mojo::matrix transform(const mojo::matrix in, const int x_center, const int y_center, int out_dim, float theta = 0, float scale = 1.f);
#endif




#ifdef MOJO_PROFILE_LAYERS
#ifdef _WIN32
	//* used for profiling layers
	double PCFreq = 0.0;
	__int64 CounterStart = 0;

	void StartCounter()
	{
		LARGE_INTEGER li;
		if (!QueryPerformanceFrequency(&li)) return;
		PCFreq = double(li.QuadPart) / 1000.0;
		QueryPerformanceCounter(&li);
		CounterStart = li.QuadPart;
	}
	double GetCounter()
	{
		LARGE_INTEGER li;
		QueryPerformanceCounter(&li);
		return double(li.QuadPart - CounterStart) / PCFreq;
	}
#else
	void StartCounter(){}
	double GetCounter(){return 0;}
#endif
	
#endif
	//*/

	void replace_str(std::string& str, const std::string& from, const std::string& to) {
		if (from.empty())
			return;
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
			str.replace(start_pos, from.length(), to);
			start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
		}
	}


// returns Energy (euclidian distance / 2) and max index
float match_labels(const float *out, const float *target, const int size, int *best_index = NULL)
{
	float E = 0;
	int max_j = 0;
	for (int j = 0; j<size; j++)
	{
		E += (out[j] - target[j])*(out[j] - target[j]);
		if (out[max_j]<out[j]) max_j = j;
	}
	if (best_index) *best_index = max_j;
	E *= 0.5;
	return E;
}
// returns index of highest value (argmax)
int arg_max(const float *out, const int size)
{
	int max_j = 0;
	for (int j = 0; j<size; j++) 
		if (out[max_j]<out[j])  
		{max_j = j; }//std::cout <<j<<",";}
	return max_j;
}

//----------------------------------------------------------------------
//  network  
//  - class that holds all the layers and connection information
//	- runs forward prediction

class network
{
	
	int _size;  // output size
	int _thread_count; // determines number of layer sets (copys of layers)
	int _internal_thread_count; // used for speeding up convolutions, etc..
	static const int MAIN_LAYER_SET = 0;

	// training related stuff
	int _batch_size;   // determines number of dW sets 
	float _skip_energy_level;
	bool _smart_train;
	std::vector <float> _running_E;
	double _running_sum_E;
	cost_function *_cost_function;
	solver *_solver;
	static const unsigned char BATCH_RESERVED = 1, BATCH_FREE = 0, BATCH_COMPLETE = 2;
	static const int BATCH_FILLED_COMPLETE = -2, BATCH_FILLED_IN_PROCESS = -1;
#ifdef MOJO_OMP
	omp_lock_t _lock_batch;
	void lock_batch() {omp_set_lock(&_lock_batch);}
	void unlock_batch() {omp_unset_lock(&_lock_batch);}
	void init_lock() {omp_init_lock(&_lock_batch);}
	void destroy_lock() {omp_destroy_lock(&_lock_batch);}
	int get_thread_num() {return omp_get_thread_num();}
#else
	void lock_batch() {}
	void unlock_batch() {}
	void init_lock(){}
	void destroy_lock() {}
	int get_thread_num() {return 0;}
#endif

public:	
	// training progress stuff
	int train_correct;
	int train_skipped;
	int stuck_counter;
	int train_updates;
	int train_samples;
	int epoch_count;
	int max_epochs;
	float best_estimated_accuracy;
	int best_accuracy_count;
	float old_estimated_accuracy;
	float estimated_accuracy;
// data augmentation stuff
	int use_augmentation; // 0=off, 1=mojo, 2=opencv
	int augment_x, augment_y;
	int augment_h_flip, augment_v_flip;
	mojo::pad_type augment_pad;
	float augment_theta;
	float augment_scale;



	// here we have multiple sets of the layers to allow threading and batch processing
	// a separate layer set is needed for each independent thread
	std::vector< std::vector<base_layer *>> layer_sets;
	
	std::map<std::string, int> layer_map;  // name-to-index of layer for layer management
	std::vector<std::pair<std::string, std::string>> layer_graph; // pairs of names of layers that are connected
	std::vector<matrix *> W; // these are the weights between/connecting layers 

	// these sets are needed because we need copies for each item in mini-batch
	std::vector< std::vector<matrix>> dW_sets; // only for training, will have _batch_size of these
	std::vector< std::vector<matrix>> dbias_sets; // only for training, will have _batch_size of these
	std::vector< unsigned char > batch_open; // only for training, will have _batch_size of these	
	

	network(const char* opt_name=NULL): _thread_count(1), _skip_energy_level(0.f), _batch_size(1) 
	{ 
		_internal_thread_count=1;
		_size=0;  
		_solver = new_solver(opt_name);
		_cost_function = NULL;
		//std::vector<base_layer *> layer_set;
		//layer_sets.push_back(layer_set);
		layer_sets.resize(1);
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size);
		_running_sum_E = 0.;
		train_correct = 0;
		train_samples = 0;
		train_skipped = 0;
		epoch_count = 0; 
		max_epochs = 1000;
		train_updates = 0;
		estimated_accuracy = 0;
		old_estimated_accuracy = 0;
		stuck_counter = 0;
		best_estimated_accuracy=0;
		best_accuracy_count=0;
		use_augmentation=0;
		augment_x = 0; augment_y = 0; augment_h_flip = 0; augment_v_flip = 0; 
		augment_pad =mojo::edge; 
		augment_theta=0; augment_scale=0;

		init_lock();
#ifdef USE_AF
		af::setDevice(0);
        af::info();
#endif
	}
	
	~network() 
	{
		clear();
		if (_cost_function) delete _cost_function;
		if(_solver) delete _solver; 
		destroy_lock();	
	}

	// call clear if you want to load a different configuration/model
	void clear()
	{
		for(int i=0; i<(int)layer_sets.size(); i++)
		{
			__for__(auto l __in__ layer_sets[i]) delete l;
			layer_sets.clear();
		}
		layer_sets.clear();
		__for__(auto w __in__ W) if(w) delete w;  
		W.clear();
		layer_map.clear();
		layer_graph.clear();
	}

	// output size of final layer;
	int out_size() {return _size;}
	
		// get input size 
	bool get_input_size(int *w, int *h, int *c)
	{
		if(layer_sets[MAIN_LAYER_SET].size()<1) return false; 
		*w=layer_sets[MAIN_LAYER_SET][0]->node.cols;*h=layer_sets[MAIN_LAYER_SET][0]->node.rows;*c=layer_sets[MAIN_LAYER_SET][0]->node.chans;
		return true;
	}

	// sets up number of layer copies to run over multiple threads
	void build_layer_sets()
	{
		int layer_cnt = (int)layer_sets.size();
		if (layer_cnt<_thread_count) layer_sets.resize(_thread_count);
		// ToDo: add shrink back /  else if(layer_cnt>_thread_count)
		sync_layer_sets();
	}

	inline int get_thread_count() {return _thread_count;}
	// must call this with max thread count before constructing layers
	// value <1 will result in thread count = # cores (including hyperthreaded)
	// must call this with max thread count before constructing layers
	// value <1 will result in thread count = # cores (including hyperthreaded)
	void enable_external_threads(int threads = -1)
	{
#ifdef MOJO_OMP
		if (threads < 1) threads = omp_get_num_procs();
		_thread_count = threads;
		if(_internal_thread_count<=_thread_count) omp_set_num_threads(_thread_count);
		omp_set_nested(1);
#else
		if (threads < 1) _thread_count = 1;
		else _thread_count = threads;
		if (threads > 1) bail("must define MOJO_OMP to used threading");
#endif
		build_layer_sets();
	}

	void enable_internal_threads(int threads = -1)
	{
#ifdef MOJO_OMP
		if (threads < 1) {threads = omp_get_num_procs(); threads = threads-1;} // one less than core count
		if(threads<1) _internal_thread_count=1;
		else _internal_thread_count=threads;
		omp_set_nested(1);
#else
		_internal_thread_count=1;
#endif

	}

	// when using threads, need to get bias data synched between all layer sets, 
	// call this after bias update in main layer set to copy the bias to the other sets
	void sync_layer_sets()
	{
		for(int i=1; i<(int)layer_sets.size();i++)
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				for(int k=0; k<layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++) 
					(layer_sets[i])[j]->bias.x[k]=(layer_sets[MAIN_LAYER_SET])[j]->bias.x[k];
	}
	
	// used to add some noise to weights
	void heat_weights()
	{
		__for__(auto w __in__ W)
		{
			if (!w) continue;
			matrix noise(w->cols, w->rows, w->chans);
			noise.fill_random_normal(1.f/ noise.size());
			//noise *= *w;
			*w += noise; 
		}
	}

	// used to add some noise to weights
	void remove_means()
	{
		__for__(auto w __in__ W)
			if(w) w->remove_mean();
	}

	// used to push a layer back in the ORDERED list of layers
	// if connect_all() is used, then the order of the push_back is used to connect the layers
	// when forward or backward propogation, this order is used for the serialized order of calculations 
	// Layer_name must be unique.
	bool push_back(const char *layer_name, const char *layer_config)
	{
		if(layer_map[layer_name]) return false; //already exists
		base_layer *l=new_layer(layer_name, layer_config);
		// set map to index

		// make sure there is a 'set' to add layers to
		if(layer_sets.size()<1)
		{
			std::vector<base_layer *> layer_set;
			layer_sets.push_back(layer_set);
		}
		// make sure layer_sets are created
		build_layer_sets();

		layer_map[layer_name] = (int)layer_sets[MAIN_LAYER_SET].size();
		layer_sets[MAIN_LAYER_SET].push_back(l);
		// upadate as potential last layer - so it sets the out size
		_size=l->fan_size();
		// add other copies needed for threading
		for(int i=1; i<(int)layer_sets.size();i++) layer_sets[i].push_back(new_layer(layer_name, layer_config));
		return true;
	}
	
	// connect 2 layers together and initialize weights
	// top and bottom concepts are reversed from literature
	// my 'top' is the input of a forward() pass and the 'bottom' is the output
	// perhaps 'top' traditionally comes from the brain model, but my 'top' comes
	// from reading order (information flows top to bottom)
	void connect(const char *layer_name_top, const char *layer_name_bottom) 
	{
		size_t i_top=layer_map[layer_name_top];
		size_t i_bottom=layer_map[layer_name_bottom];

		base_layer *l_top= layer_sets[MAIN_LAYER_SET][i_top];
		base_layer *l_bottom= layer_sets[MAIN_LAYER_SET][i_bottom];
		
		int w_i=(int)W.size();
//		matrix *w = l_bottom->new_connection(*l_top, w_i);
        enclave_matrix *w = l_bottom->new_connection(*l_top, w_i);
		
		printf("new connection finished..\n");
		W.push_back(w);
		printf("pushed back the new matrix\n");
		layer_graph.push_back(std::make_pair(layer_name_top,layer_name_bottom));
		printf("pushed back the layer_graph\n");
		// need to build connections for other batches/threads
		for(int i=1; i<(int)layer_sets.size(); i++)
		{
		    printf("delete new_connection\n");
			l_top= layer_sets[i][i_top];
			l_bottom= layer_sets[i][i_bottom];
			delete l_bottom->new_connection(*l_top, w_i);
			printf("new connection finished..\n");
		}
		printf("begin preparing the solver\n");
		// we need to let solver prepare space for stateful information 
		if (_solver)
		{
			if (w)_solver->push_back(w->cols, w->rows, w->chans);
			else _solver->push_back(1, 1, 1);
		}

		int fan_in=l_bottom->fan_size();
		int fan_out=l_top->fan_size();

		// ToDo: this may be broke when 2 layers connect to one. need to fix (i.e. resnet)
		// after all connections, run through and do weights with correct fan count
		printf("begin initialize weights\n");
		// initialize weights - ToDo: separate and allow users to configure(?)
		if (w && l_bottom->has_weights())
		{
		    printf("has_weights.\n");
			if (strcmp(l_bottom->p_act->name, "tanh") == 0)
			{
			    printf("fill_random_uniform begin: tanh.\n");
				// xavier : for tanh
				float weight_base = (float)(std::sqrt(6. / ((double)fan_in + (double)fan_out)));
				//		float weight_base = (float)(std::sqrt(.25/( (double)fan_in)));
				w->fill_random_uniform(weight_base);
				printf("fill_random_uniform finished.\n");
			}
			else if ((strcmp(l_bottom->p_act->name, "sigmoid") == 0) || (strcmp(l_bottom->p_act->name, "sigmoid") == 0))
			{
			    printf("fill_random_uniform begin: sigmoid.\n");
				// xavier : for sigmoid
				float weight_base = 4.f*(float)(std::sqrt(6. / ((double)fan_in + (double)fan_out)));
				w->fill_random_uniform(weight_base);
				printf("fill_random_uniform finished.\n");
			}
			else if ((strcmp(l_bottom->p_act->name, "lrelu") == 0) || (strcmp(l_bottom->p_act->name, "relu") == 0)
				|| (strcmp(l_bottom->p_act->name, "vlrelu") == 0) || (strcmp(l_bottom->p_act->name, "elu") == 0))
			{
				// he : for relu
				printf("fill_random_uniform begin: relu.\n");
				float weight_base = (float)(std::sqrt(2. / (double)fan_in));
				w->fill_random_normal(weight_base);
				printf("fill_random_uniform finished.\n");
			}
			else
			{
				// lecun : orig
				printf("fill_random_uniform begin: lecun : orig.\n");
				float weight_base = (float)(std::sqrt(1. / (double)fan_in));
				w->fill_random_uniform(weight_base);
				
				printf("fill_random_uniform finished.\n");
			}
		}
		else if (w) w->fill(0);
		printf("finished initialize weights\n");
	}

	// automatically connect all layers in the order they were provided 
	// easy way to go, but can't deal with branch/highway/resnet/inception types of architectures
	void connect_all()
	{	
		for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size()-1; j++) 
			connect(layer_sets[MAIN_LAYER_SET][j]->name.c_str(), layer_sets[MAIN_LAYER_SET][j+1]->name.c_str());
	}

	int get_layer_index(const char *name)
	{
		for (int j = 0; j < (int)layer_sets[MAIN_LAYER_SET].size(); j++)
			if (layer_sets[MAIN_LAYER_SET][j]->name.compare(name) == 0)
				return j;
		return -1;
	}
	
	// get the list of layers used (but not connection information)
	std::string get_configuration()
	{
		std::string str;
		std::string space("  ");
		std::string symbol(" : ");
		
		// print all layer configs
		for (int j = 0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++) 
		{
		    //std::string jstr(dtoa(j));
		    std::string jstr;
        if(j == 0) jstr = "0";
        else        jstr = dtoa(j);
  

		    std::string lname(layer_sets[MAIN_LAYER_SET][j]->name);
		    std::string lsets(layer_sets[MAIN_LAYER_SET][j]->get_config_string());
		    //str+= "  "+ std::to_string((long long)j) +" : " +layer_sets[MAIN_LAYER_SET][j]->name +" : " + layer_sets[MAIN_LAYER_SET][j]->get_config_string();
		    str += space + jstr + symbol + lname + symbol + lsets;
		}
		
		str += "\n";
		// print layer links
		if (layer_graph.size() <= 0) return str;
		
		for (int j = 0; j < (int)layer_graph.size(); j++)
		{
			if (j % 3 == 0) str += "  ";
			if((j % 3 == 1)|| (j % 3 == 2)) str += ", ";
			str +=layer_graph[j].first + "-" + layer_graph[j].second;
			if (j % 3 == 2) str += "\n";
		}
		return str;
	}
	
	// performs forward pass and returns class index
	// do not delete or modify the returned pointer. it is a live pointer to the last layer in the network
	// if calling over multiple threads, provide the thread index since the interal data is not otherwise thread safe
	int predict_class(const float *in, int _thread_number = -1)
	{
		const float* out = forward(in, _thread_number);
//		for(int i = 0; i < out_size(); i++)
//		    printf("%d: %f\n", i, out[i]);
		return arg_max(out, out_size());
	}

	//----------------------------------------------------------------------------------------------------------
	// F O R W A R D
	//
	// the main forward pass 
	// if calling over multiple threads, provide the thread index since the interal data is not otherwise thread safe
	// train parameter is used to designate the forward pass is used in training (it turns on dropout layers, etc..)
	float* forward(const float *in, int _thread_number=-1, int _train=0)
	{
//        for(int i = W[0]->size()-10; i < W[0]->size(); i++)
//            printf("W[i]->x[%d] = %f\n", i, W[0]->x[i]);
		if(_thread_number<0) _thread_number=get_thread_num();
		if (_thread_number > _thread_count && _thread_count>0) bail("need to enable threading\n");
		if (_thread_number >= (int)layer_sets.size()) bail("need to enable threading\n");

		//std::cout << get_thread_num() << ",";
		// clear nodes to zero & find input layers
		std::vector<base_layer *> inputs;
		__for__(auto layer __in__ layer_sets[_thread_number])
		{
			if (dynamic_cast<input_layer*> (layer) != NULL)  inputs.push_back(layer);
			layer->set_threading(_internal_thread_count);
			layer->node.fill(0.f);
		}
		// first layer assumed input. copy input to it 
		const float *in_ptr = in;
		//base_layer * layer = layer_sets[_thread_number][0];

		//memcpy(layer->node.x, in, sizeof(float)*layer->node.size());
		
		__for__(auto layer __in__ inputs)
		{
			memcpy(layer->node.x, in_ptr, sizeof(float)*layer->node.size());
			in_ptr += layer->node.size();
		}
		//for (int i = 0; i < layer->node.size(); i++)
		//	layer_sets[_thread_number][0]->node.x[i] = in[i];
		// for all layers
		__for__(auto layer __in__ layer_sets[_thread_number])
		{
			// add bias and activate these outputs (they should all be summed up from other branches at this point)
			//for(int j=0; j<layer->node.chans; j+=10) for (int i=0; i<layer->node.cols*layer->node.rows; i+=10)	std::cout<< layer->node.x[i+j*layer->node.chan_stride] <<"|";
			layer->activate_nodes(); 
			
			//for(int j=0; j<layer->node.chans; j++) for (int i=0; i<layer->node.cols*layer->node.rows; i+=10)	std::cout<< layer->node.x[i+j*layer->node.chan_stride] <<"|";
			// send output signal downstream (note in this code 'top' is input layer, 'bottom' is output - bucking tradition
			__for__ (auto &link __in__ layer->forward_linked_layers)
			{
				// instead of having a list of paired connections, just use the shape of W to determine connections
				// this is harder to read, but requires less look-ups
				// the 'link' variable is a std::pair created during the connect() call for the layers
				int connection_index = link.first; 
				base_layer *p_bottom = link.second;
				// weight distribution of the signal to layers under it

				p_bottom->accumulate_signal(*layer, *W[connection_index], _train);
				//if (p_bottom->has_weights())
			//for(int j=0; j<layer->node.chans; j++) 
			//int j=0;	for (int i=0; i<layer->node.cols*layer->node.rows; i+=10)	std::cout<< layer->node.x[i+j*layer->node.chan_stride] <<"|";
	
			}

		}
		// return pointer to float * result from last layer
		return layer_sets[_thread_number][layer_sets[_thread_number].size()-1]->node.x;
	}
	
	
	void fprint_networkfile(const char *fmt, ...) { 
	    char buf[BUFSIZ] = { '\0' };
	    va_list ap;
	    va_start(ap, fmt);
	    vsnprintf(buf, BUFSIZ, fmt, ap);
	    va_end(ap);
	    ocall_fprint_networkfile(buf);
    }

	//----------------------------------------------------------------------------------------------------------
	// W R I T E
	//
	// write parameters to stream/file
	// note that this does not persist intermediate training information that could be needed to 'pickup where you left off'
	bool write(char *filename, bool binary = false, bool final = false) 
	{
	    int retocall;
	    open_outputnetworkfile(&retocall, filename);
	    
	    
	    // save layers
		int layer_cnt = (int)layer_sets[MAIN_LAYER_SET].size();
//		int ignore_cnt = 0;
//		for (int j = 0; j<(int)layer_sets[0].size(); j++)
//			if (dynamic_cast<dropout_layer*> (layer_sets[0][j]) != NULL)  ignore_cnt++;
        
        
        fprint_networkfile("mojo01\n");
        fprint_networkfile("%d\n", (int)(layer_cnt));        
//		ofs<<"mojo01" << std::endl;
//		ofs<<(int)(layer_cnt)<<std::endl;
		
		for(int j=0; j<(int)layer_sets[0].size(); j++)
		    fprint_networkfile("%s\n%s", layer_sets[MAIN_LAYER_SET][j]->name, layer_sets[MAIN_LAYER_SET][j]->get_config_string().c_str());
		//	ofs << layer_sets[MAIN_LAYER_SET][j]->name << std::endl << layer_sets[MAIN_LAYER_SET][j]->get_config_string();	
			
//			if (dynamic_cast<dropout_layer*> (layer_sets[0][j]) != NULL)

		// save graph
		fprint_networkfile("%d\n", (int)layer_graph.size());     
		//ofs<<(int)layer_graph.size()<<std::endl;
		for(int j=0; j<(int)layer_graph.size(); j++)
		    fprint_networkfile("%s\n%s\n", layer_graph[j].first.c_str(), layer_graph[j].second.c_str());
		//	ofs<<layer_graph[j].first << std::endl << layer_graph[j].second << std::endl;

		if(binary)
		{
			//ofs<<(int)1<<std::endl; // flags that this is binary data
			fprint_networkfile("1\n");
			// binary version to save space if needed
			// save bias info
			
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				if(layer_sets[MAIN_LAYER_SET][j]->use_bias())
        {
            int breakdown = 0; // 
            while(breakdown + blocksize < layer_sets[MAIN_LAYER_SET][j]->bias.size())
            {
				      ocall_write((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x + breakdown*sizeof(float), blocksize*sizeof(float));
              breakdown += blocksize;
            }
            ocall_write((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x + breakdown*sizeof(float), (layer_sets[MAIN_LAYER_SET][j]->bias.size()-breakdown)*sizeof(float));
				//    ocall_write((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
        }
				//    for(int k = 0; k < layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float); k++)
				//	    fprint_networkfile("%c", (char*)layer_sets[MAIN_LAYER_SET][j]->bias.x+k);
					//ofs.write((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
			// save weights
			for (int j = 0; j < (int)W.size(); j++)
			{
				if (W[j])
        {
            int breakdown = 0; // 
            while(breakdown + blocksize < W[j]->size())
            {
				      ocall_write((char*)W[j]->x + breakdown*sizeof(float), blocksize*sizeof(float));
              breakdown += blocksize;
            }
            ocall_write((char*)W[j]->x + breakdown*sizeof(float), (W[j]->size()-breakdown)*sizeof(float));
        }
				//    for(int k = 0; k < W[j]->size()*sizeof(float); k++)
				//        fprint_networkfile("%c", (char*)W[j]->x+k);
				//	ofs.write((char*)W[j]->x, W[j]->size()*sizeof(float));
			}
		}
		else
		{
			//ofs<<(int)0<<std::endl;
			fprint_networkfile("0\n");
			// save bias info
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
			{
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
				{
					for (int k = 0; k < layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)  
					    fprint_networkfile("%f ", layer_sets[MAIN_LAYER_SET][j]->bias.x[k]);
					    //ofs << layer_sets[MAIN_LAYER_SET][j]->bias.x[k] << " ";
					fprint_networkfile("\n");
					//ofs << std::endl;
				}
			}
			// save weights
			for(int j=0; j<(int)W.size(); j++)
			{
				if (W[j])
				{
					for (int i = 0; i < W[j]->size(); i++) 
					    fprint_networkfile("%f ", W[j]->x[i]);
					    //ofs << W[j]->x[i] << " ";
					fprint_networkfile("\n");
					//ofs << std::endl;
				}
			}
		}
		//ofs.flush();
		close_outputnetworkfile();
		return true;
	}
	
	// read network from a file/stream
	
	bool endoffile;
	std::string getcleanline()
	{
		std::string s;

		// The characters in the stream are read one-by-one using a std::streambuf.
		// That is faster than reading them one-by-one using the std::istream.
		// Code that uses streambuf this way must be guarded by a sentry object.
		// The sentry object performs various tasks,
		// such as thread synchronization and updating the stream state.

		//std::istream::sentry se(ifs, true);
		//std::streambuf* sb = ifs.rdbuf();

		for (;;) {
			char c;

			ocall_fread_networkfile(&c);//sb->sbumpc();
			//printf("%d\n", c);
			switch (c) {
			case '\n':
			    //printf("s = %s\n", s);
				return s;
			case '\r':
				//if (sb->sgetc() == '\n') sb->sbumpc();
				char cc; ocall_fread_networkfile(&cc);
				//printf("\\r got, %d\n", c, cc);
				if (cc == '\n') 
				    return s;
			case EOF:
			    endoffile = true;
			    //printf("end of file, %d, %s\n", c, s);
				// Also handle the case when the last line has no line ending
				if (s.empty()) //ifs.setstate(std::ios::eofbit);
				    return s;
			default:
				s += (char)c;
			}
		}
	}
	
	//----------------------------------------------------------------------------------------------------------
	// R E A D
	//
	bool read()
	{
//		if(!ifs.good()) return false;
		std::string s;
		s = getcleanline();
		int layer_count;
		int version = 0;
		if (s.compare("mojo01")==0)
		{
			s = getcleanline();
			layer_count = atoi(s.c_str());
			version = 1;
			
			printf("version = 1, layer_count: %d, line: %s\n", layer_count, s);
		}
		else if (s.find("mojo:") == 0)
		{
		    //printf("version = -1\n");
			version = -1;
			int cnt = 1;

			while (!endoffile)
			{
				s = getcleanline();
				if (s.empty()) continue;
				if(s[0]=='#') continue;
                
				push_back(dtoa(cnt), s.c_str());
				
				printf("layer %d: %s\n", cnt, s.c_str());
				cnt++;
			}
			
			connect_all();

			// copies batch=0 stuff to other batches
			sync_layer_sets();
			return true;
		}
		else
		{
			layer_count = atoi(s.c_str());
			
			printf("layer_count: %d, line: %s\n", layer_count, s);
		}
		// read layer def
		std::string layer_name;
		std::string layer_def;
		for (auto i=0; i<layer_count; i++)
		{
			layer_name = getcleanline();
			layer_def = getcleanline();
			push_back(layer_name.c_str(), layer_def.c_str());
			
			printf("%s: %s\n", layer_name.c_str(), layer_def.c_str());
		}

		// read graph
		int graph_count;
		//ifs>>graph_count;
		ocall_getint(&graph_count);
		
		end_this_line();
		
		//printf("graph_count: %d\n", graph_count);
		//getline(ifs,s); // get endline; just want to end reading the line? the result is not important
		if (graph_count <= 0)
		{
			connect_all();
		}
		else
	    {
		    std::string layer_name1;
		    std::string layer_name2;
		    for (auto i=0; i<graph_count; i++)
		    {
			    layer_name1= getcleanline();
			   
			    layer_name2 = getcleanline();
			    
			    printf("%d: %s", i, layer_name1.c_str());
			    printf("\t%s", layer_name2.c_str());
			    
			    printf("\n");
			    
			    connect(layer_name1.c_str(), layer_name2.c_str());
		    }
		}

		int binary;
		s=getcleanline(); // get endline
		binary = atoi(s.c_str());
		
		printf("binary: %d\n", binary);

		// binary version to save space if needed
		if(binary==1)
		{
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
				{
  					//int c = layer_sets[MAIN_LAYER_SET][j]->bias.chans;
  					//int cs = layer_sets[MAIN_LAYER_SET][j]->bias.chan_stride  					
  					// use ocall_read instead, ww31
  				//	ocall_read((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
                    int breakdown = 0; // 
                    while(breakdown + blocksize < layer_sets[MAIN_LAYER_SET][j]->bias.size())
                    {
                        ocall_read((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x + breakdown*sizeof(float), blocksize*sizeof(float));
                        breakdown += blocksize;
                    }
                    ocall_read((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x + breakdown*sizeof(float), (layer_sets[MAIN_LAYER_SET][j]->bias.size()-breakdown)*sizeof(float));
  					
  					//	ifs.read((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
				}
			for (int j = 0; j < (int)W.size(); j++)
			{
                printf("loading weight for %d-th layer\n", j);
				if (W[j])
				{
				    //   ocall_read_outenclave((uint64_t)(W[j]->x), W[j]->size()*sizeof(float));
                    ocall_read_outenclave((uint64_t)(W[j]), W[j]->size()*sizeof(float));
                /*    int breakdown = 0; // 
                    while(breakdown + blocksize < W[j]->size())
                    {
                        ocall_read((char*)W[j]->x + breakdown*sizeof(float), blocksize*sizeof(float));
                        breakdown += blocksize;
                    }
                    ocall_read((char*)W[j]->x + breakdown*sizeof(float), (W[j]->size()-breakdown)*sizeof(float));
				//	ifs.read((char*)W[j]->x, W[j]->size()*sizeof(float));*/
				}
			}
		}
		else if(binary==0)// text version
		{
			// read bias
			for(int j=0; j<layer_count; j++)
			{
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
				{
				//	int c = layer_sets[MAIN_LAYER_SET][j]->bias.chans;
				//	int cs = layer_sets[MAIN_LAYER_SET][j]->bias.chan_stride;

				//	for (int i = 0; i < c; i++)
					for (int k = 0; k < layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)
					{
						//ifs >> layer_sets[MAIN_LAYER_SET][j]->bias.x[k];
						
						ocall_getfloat(&layer_sets[MAIN_LAYER_SET][j]->bias.x[k]);
						
						//std::cout << layer_sets[MAIN_LAYER_SET][j]->bias.x[k] << ",";
					}
					//ifs.ignore();// getline(ifs, s); // get endline
					end_this_line();
				}
			}

			// read weights
			for (auto j=0; j<(int)W.size(); j++)
			{
				if (W[j])
				{
					for (int i = 0; i < W[j]->size(); i++) 
					    ocall_getfloat(&W[j]->x[i]);
					    //ifs >> W[j]->x[i];
					//ifs.ignore(); //getline(ifs, s); // get endline
					end_this_line();
				}
			}
		}
	
		// copies batch=0 stuff to other batches
		sync_layer_sets();

		return true;
	}
	bool read(char *filename)
	{
		//std::ifstream fs(filename.c_str(),std::ios::binary);
		int retocall;
	    open_networkfile(&retocall, filename);
		if (retocall == 0)
		{
		    endoffile = false;
			bool ret = read();
			close_networkfile();
			return ret;
		}
		else return false;
	}
//	bool read(const char *filename) { return  read(std::string(filename)); }
	
	float get_learning_rate() {return 0;}
	void set_learning_rate(float alpha) {}
	void train(float *in, float *target){}
	void reset() {}
	float get_smart_train_level() {return 0;}
	void set_smart_train_level(float _level) {}
	bool get_smart_train() { return false; }
	void set_smart_train(bool _use) {}
};

}
