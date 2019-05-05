#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_generate_random_number_t {
	int ms_retval;
} ms_generate_random_number_t;

typedef struct ms_new_network_t {
	char* ms_str;
} ms_new_network_t;

typedef struct ms_cnn_outsize_t {
	int ms_retval;
} ms_cnn_outsize_t;

typedef struct ms_classification_t {
	int ms_retval;
	float* ms_testimage;
	int ms_cnt;
} ms_classification_t;

typedef struct ms_get_estimated_accuracy_t {
	float ms_retval;
} ms_get_estimated_accuracy_t;

typedef struct ms_ocall_print_t {
	char* ms_str;
} ms_ocall_print_t;

typedef struct ms_open_file_t {
	char* ms_str;
} ms_open_file_t;


typedef struct ms_read_file_t {
	char* ms_dest;
	int ms_sz;
} ms_read_file_t;

typedef struct ms_mojo_sleep_t {
	unsigned int ms_milliseconds;
} ms_mojo_sleep_t;

typedef struct ms_open_networkfile_t {
	int ms_retval;
	char* ms_str;
} ms_open_networkfile_t;

typedef struct ms_open_outputnetworkfile_t {
	int ms_retval;
	char* ms_str;
} ms_open_outputnetworkfile_t;

typedef struct ms_ocall_fprint_networkfile_t {
	char* ms_str;
} ms_ocall_fprint_networkfile_t;

typedef struct ms_ocall_fread_networkfile_t {
	char ms_retval;
} ms_ocall_fread_networkfile_t;

typedef struct ms_ocall_getint_t {
	int ms_retval;
} ms_ocall_getint_t;

typedef struct ms_ocall_getfloat_t {
	float ms_retval;
} ms_ocall_getfloat_t;

typedef struct ms_ocall_read_t {
	char* ms_src;
	int ms_sz;
} ms_ocall_read_t;

typedef struct ms_ocall_read_outenclave_t {
	uint64_t ms_src;
	int ms_sz;
} ms_ocall_read_outenclave_t;

typedef struct ms_ocall_write_t {
	char* ms_src;
	int ms_sz;
} ms_ocall_write_t;




typedef struct ms_ocall_newmatrix_t {
	uint64_t ms_retval;
	uint64_t* ms_px;
	int* ms_size;
	int ms_cols;
	int ms_rows;
	int ms_chans;
} ms_ocall_newmatrix_t;

typedef struct ms_ocall_fill_uniform_t {
	uint64_t ms_pmatrix;
	float ms_range;
} ms_ocall_fill_uniform_t;

typedef struct ms_ocall_fill_normal_t {
	uint64_t ms_pmatrix;
	float ms_std;
} ms_ocall_fill_normal_t;

static sgx_status_t SGX_CDECL Enclave_ocall_print(void* pms)
{
	ms_ocall_print_t* ms = SGX_CAST(ms_ocall_print_t*, pms);
	ocall_print((const char*)ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_open_file(void* pms)
{
	ms_open_file_t* ms = SGX_CAST(ms_open_file_t*, pms);
	open_file((const char*)ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_close_file(void* pms)
{
	if (pms != NULL) return SGX_ERROR_INVALID_PARAMETER;
	close_file();
	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_read_file(void* pms)
{
	ms_read_file_t* ms = SGX_CAST(ms_read_file_t*, pms);
	read_file(ms->ms_dest, ms->ms_sz);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_mojo_sleep(void* pms)
{
	ms_mojo_sleep_t* ms = SGX_CAST(ms_mojo_sleep_t*, pms);
	mojo_sleep(ms->ms_milliseconds);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_open_networkfile(void* pms)
{
	ms_open_networkfile_t* ms = SGX_CAST(ms_open_networkfile_t*, pms);
	ms->ms_retval = open_networkfile((const char*)ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_open_outputnetworkfile(void* pms)
{
	ms_open_outputnetworkfile_t* ms = SGX_CAST(ms_open_outputnetworkfile_t*, pms);
	ms->ms_retval = open_outputnetworkfile((const char*)ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_fprint_networkfile(void* pms)
{
	ms_ocall_fprint_networkfile_t* ms = SGX_CAST(ms_ocall_fprint_networkfile_t*, pms);
	ocall_fprint_networkfile((const char*)ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_fread_networkfile(void* pms)
{
	ms_ocall_fread_networkfile_t* ms = SGX_CAST(ms_ocall_fread_networkfile_t*, pms);
	ms->ms_retval = ocall_fread_networkfile();

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_getint(void* pms)
{
	ms_ocall_getint_t* ms = SGX_CAST(ms_ocall_getint_t*, pms);
	ms->ms_retval = ocall_getint();

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_getfloat(void* pms)
{
	ms_ocall_getfloat_t* ms = SGX_CAST(ms_ocall_getfloat_t*, pms);
	ms->ms_retval = ocall_getfloat();

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_read(void* pms)
{
	ms_ocall_read_t* ms = SGX_CAST(ms_ocall_read_t*, pms);
	ocall_read(ms->ms_src, ms->ms_sz);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_read_outenclave(void* pms)
{
	ms_ocall_read_outenclave_t* ms = SGX_CAST(ms_ocall_read_outenclave_t*, pms);
	ocall_read_outenclave(ms->ms_src, ms->ms_sz);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_write(void* pms)
{
	ms_ocall_write_t* ms = SGX_CAST(ms_ocall_write_t*, pms);
	ocall_write(ms->ms_src, ms->ms_sz);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_end_this_line(void* pms)
{
	if (pms != NULL) return SGX_ERROR_INVALID_PARAMETER;
	end_this_line();
	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_close_networkfile(void* pms)
{
	if (pms != NULL) return SGX_ERROR_INVALID_PARAMETER;
	close_networkfile();
	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_close_outputnetworkfile(void* pms)
{
	if (pms != NULL) return SGX_ERROR_INVALID_PARAMETER;
	close_outputnetworkfile();
	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_newmatrix(void* pms)
{
	ms_ocall_newmatrix_t* ms = SGX_CAST(ms_ocall_newmatrix_t*, pms);
	ms->ms_retval = ocall_newmatrix(ms->ms_px, ms->ms_size, ms->ms_cols, ms->ms_rows, ms->ms_chans);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_fill_uniform(void* pms)
{
	ms_ocall_fill_uniform_t* ms = SGX_CAST(ms_ocall_fill_uniform_t*, pms);
	ocall_fill_uniform(ms->ms_pmatrix, ms->ms_range);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_fill_normal(void* pms)
{
	ms_ocall_fill_normal_t* ms = SGX_CAST(ms_ocall_fill_normal_t*, pms);
	ocall_fill_normal(ms->ms_pmatrix, ms->ms_std);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[20];
} ocall_table_Enclave = {
	20,
	{
		(void*)Enclave_ocall_print,
		(void*)Enclave_open_file,
		(void*)Enclave_close_file,
		(void*)Enclave_read_file,
		(void*)Enclave_mojo_sleep,
		(void*)Enclave_open_networkfile,
		(void*)Enclave_open_outputnetworkfile,
		(void*)Enclave_ocall_fprint_networkfile,
		(void*)Enclave_ocall_fread_networkfile,
		(void*)Enclave_ocall_getint,
		(void*)Enclave_ocall_getfloat,
		(void*)Enclave_ocall_read,
		(void*)Enclave_ocall_read_outenclave,
		(void*)Enclave_ocall_write,
		(void*)Enclave_end_this_line,
		(void*)Enclave_close_networkfile,
		(void*)Enclave_close_outputnetworkfile,
		(void*)Enclave_ocall_newmatrix,
		(void*)Enclave_ocall_fill_uniform,
		(void*)Enclave_ocall_fill_normal,
	}
};
sgx_status_t generate_random_number(sgx_enclave_id_t eid, int* retval)
{
	sgx_status_t status;
	ms_generate_random_number_t ms;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t new_network(sgx_enclave_id_t eid, const char* str)
{
	sgx_status_t status;
	ms_new_network_t ms;
	ms.ms_str = (char*)str;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t cnn_outsize(sgx_enclave_id_t eid, int* retval)
{
	sgx_status_t status;
	ms_cnn_outsize_t ms;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t classification(sgx_enclave_id_t eid, int* retval, float* testimage, int cnt)
{
	sgx_status_t status;
	ms_classification_t ms;
	ms.ms_testimage = testimage;
	ms.ms_cnt = cnt;
	status = sgx_ecall(eid, 3, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t get_estimated_accuracy(sgx_enclave_id_t eid, float* retval)
{
	sgx_status_t status;
	ms_get_estimated_accuracy_t ms;
	status = sgx_ecall(eid, 4, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

