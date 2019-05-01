#ifndef ENCLAVE_U_H__
#define ENCLAVE_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "sgx_edger8r.h" /* for sgx_satus_t etc. */

#include "../Enclave/msse2/emmintrin.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_print, (const char* str));
void SGX_UBRIDGE(SGX_NOCONVENTION, open_file, (const char* str));
void SGX_UBRIDGE(SGX_NOCONVENTION, close_file, ());
void SGX_UBRIDGE(SGX_NOCONVENTION, read_file, (char* dest, int sz));
void SGX_UBRIDGE(SGX_NOCONVENTION, mojo_sleep, (unsigned int milliseconds));
int SGX_UBRIDGE(SGX_NOCONVENTION, open_networkfile, (const char* str));
int SGX_UBRIDGE(SGX_NOCONVENTION, open_outputnetworkfile, (const char* str));
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_fprint_networkfile, (const char* str));
char SGX_UBRIDGE(SGX_NOCONVENTION, ocall_fread_networkfile, ());
int SGX_UBRIDGE(SGX_NOCONVENTION, ocall_getint, ());
float SGX_UBRIDGE(SGX_NOCONVENTION, ocall_getfloat, ());
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_read, (char* src, int sz));
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_write, (char* src, int sz));
void SGX_UBRIDGE(SGX_NOCONVENTION, end_this_line, ());
void SGX_UBRIDGE(SGX_NOCONVENTION, close_networkfile, ());
void SGX_UBRIDGE(SGX_NOCONVENTION, close_outputnetworkfile, ());

sgx_status_t generate_random_number(sgx_enclave_id_t eid, int* retval);
sgx_status_t new_network(sgx_enclave_id_t eid, const char* str);
sgx_status_t cnn_outsize(sgx_enclave_id_t eid, int* retval);
sgx_status_t classification(sgx_enclave_id_t eid, int* retval, float* testimage, int cnt);
sgx_status_t get_epoch(sgx_enclave_id_t eid, int* retval);
sgx_status_t epoch(sgx_enclave_id_t eid, char* str);
sgx_status_t train(sgx_enclave_id_t eid, float* testimage, int cnt, int train_labels);
sgx_status_t end_epoch(sgx_enclave_id_t eid);
sgx_status_t get_estimated_accuracy(sgx_enclave_id_t eid, float* retval);
sgx_status_t reset_smart_training(sgx_enclave_id_t eid);
sgx_status_t write_model_file(sgx_enclave_id_t eid, char* model_file);
sgx_status_t elvis_left_the_building(sgx_enclave_id_t eid, int* retval);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
