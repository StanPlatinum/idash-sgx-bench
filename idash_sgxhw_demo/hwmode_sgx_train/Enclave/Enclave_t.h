#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */

#include "../Enclave/msse2/emmintrin.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif


int generate_random_number();
void new_network(const char* str);
int cnn_outsize();
int classification(float* testimage, int cnt);
int get_epoch();
void epoch(char* str);
void train(float* testimage, int cnt, int train_labels);
void end_epoch();
float get_estimated_accuracy();
void reset_smart_training();
void write_model_file(char* model_file);
int elvis_left_the_building();

sgx_status_t SGX_CDECL ocall_print(const char* str);
sgx_status_t SGX_CDECL open_file(const char* str);
sgx_status_t SGX_CDECL close_file();
sgx_status_t SGX_CDECL read_file(char* dest, int sz);
sgx_status_t SGX_CDECL mojo_sleep(unsigned int milliseconds);
sgx_status_t SGX_CDECL open_networkfile(int* retval, const char* str);
sgx_status_t SGX_CDECL open_outputnetworkfile(int* retval, const char* str);
sgx_status_t SGX_CDECL ocall_fprint_networkfile(const char* str);
sgx_status_t SGX_CDECL ocall_fread_networkfile(char* retval);
sgx_status_t SGX_CDECL ocall_getint(int* retval);
sgx_status_t SGX_CDECL ocall_getfloat(float* retval);
sgx_status_t SGX_CDECL ocall_read(char* src, int sz);
sgx_status_t SGX_CDECL ocall_write(char* src, int sz);
sgx_status_t SGX_CDECL end_this_line();
sgx_status_t SGX_CDECL close_networkfile();
sgx_status_t SGX_CDECL close_outputnetworkfile();

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
