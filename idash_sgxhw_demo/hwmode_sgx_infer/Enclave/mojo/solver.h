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
//    solver.h: stochastic optimization approaches
// ==================================================================== mojo ==

#pragma once

#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdlib>

#include "core_math.h"

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

namespace mojo {



class solver
{
public:
	// learning rates are 'tweaked' in inc_w function so that they can be similar for all solvers
	float learning_rate;
	solver(): learning_rate(0.01f) {}
	virtual ~solver(){}
	virtual void reset() {}
	// this increments the weight matrix w, which corresponds to connection index 'g'
	// bottom is the number of grads coming up from the lower layer
	// top is the current output node value of the upper layer
	virtual void increment_w(matrix *w,  int g, const matrix &dW, const float custom_factor=1.0f){}//, matrix *top){}
	virtual void push_back(int w, int h, int c){}	
};

solver* new_solver(const char *type) {return NULL;}
solver* new_solver(std::string act){return NULL;}



} // namespace
