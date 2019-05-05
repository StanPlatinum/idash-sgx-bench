#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */

#include <errno.h>
#include <string.h> /* for memcpy etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)


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

static sgx_status_t SGX_CDECL sgx_generate_random_number(void* pms)
{
	ms_generate_random_number_t* ms = SGX_CAST(ms_generate_random_number_t*, pms);
	sgx_status_t status = SGX_SUCCESS;

	CHECK_REF_POINTER(pms, sizeof(ms_generate_random_number_t));

	ms->ms_retval = generate_random_number();


	return status;
}

static sgx_status_t SGX_CDECL sgx_new_network(void* pms)
{
	ms_new_network_t* ms = SGX_CAST(ms_new_network_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_str = ms->ms_str;
	size_t _len_str = _tmp_str ? strlen(_tmp_str) + 1 : 0;
	char* _in_str = NULL;

	CHECK_REF_POINTER(pms, sizeof(ms_new_network_t));
	CHECK_UNIQUE_POINTER(_tmp_str, _len_str);

	if (_tmp_str != NULL) {
		_in_str = (char*)malloc(_len_str);
		if (_in_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_str, _tmp_str, _len_str);
		_in_str[_len_str - 1] = '\0';
	}
	new_network((const char*)_in_str);
err:
	if (_in_str) free((void*)_in_str);

	return status;
}

static sgx_status_t SGX_CDECL sgx_cnn_outsize(void* pms)
{
	ms_cnn_outsize_t* ms = SGX_CAST(ms_cnn_outsize_t*, pms);
	sgx_status_t status = SGX_SUCCESS;

	CHECK_REF_POINTER(pms, sizeof(ms_cnn_outsize_t));

	ms->ms_retval = cnn_outsize();


	return status;
}

static sgx_status_t SGX_CDECL sgx_classification(void* pms)
{
	ms_classification_t* ms = SGX_CAST(ms_classification_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_testimage = ms->ms_testimage;
	int _tmp_cnt = ms->ms_cnt;
	size_t _len_testimage = _tmp_cnt * sizeof(*_tmp_testimage);
	float* _in_testimage = NULL;

	if ((size_t)_tmp_cnt > (SIZE_MAX / sizeof(*_tmp_testimage))) {
		status = SGX_ERROR_INVALID_PARAMETER;
		goto err;
	}

	CHECK_REF_POINTER(pms, sizeof(ms_classification_t));
	CHECK_UNIQUE_POINTER(_tmp_testimage, _len_testimage);

	if (_tmp_testimage != NULL) {
		_in_testimage = (float*)malloc(_len_testimage);
		if (_in_testimage == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_testimage, _tmp_testimage, _len_testimage);
	}
	ms->ms_retval = classification(_in_testimage, _tmp_cnt);
err:
	if (_in_testimage) free(_in_testimage);

	return status;
}

static sgx_status_t SGX_CDECL sgx_get_estimated_accuracy(void* pms)
{
	ms_get_estimated_accuracy_t* ms = SGX_CAST(ms_get_estimated_accuracy_t*, pms);
	sgx_status_t status = SGX_SUCCESS;

	CHECK_REF_POINTER(pms, sizeof(ms_get_estimated_accuracy_t));

	ms->ms_retval = get_estimated_accuracy();


	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv;} ecall_table[5];
} g_ecall_table = {
	5,
	{
		{(void*)(uintptr_t)sgx_generate_random_number, 0},
		{(void*)(uintptr_t)sgx_new_network, 0},
		{(void*)(uintptr_t)sgx_cnn_outsize, 0},
		{(void*)(uintptr_t)sgx_classification, 0},
		{(void*)(uintptr_t)sgx_get_estimated_accuracy, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[20][5];
} g_dyn_entry_table = {
	20,
	{
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL ocall_print(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_print_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_print_t);
	void *__tmp = NULL;

	ocalloc_size += (str != NULL && sgx_is_within_enclave(str, _len_str)) ? _len_str : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_print_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_print_t));

	if (str != NULL && sgx_is_within_enclave(str, _len_str)) {
		ms->ms_str = (char*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_str);
		memcpy((void*)ms->ms_str, str, _len_str);
	} else if (str == NULL) {
		ms->ms_str = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	status = sgx_ocall(0, ms);


	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL open_file(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_open_file_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_open_file_t);
	void *__tmp = NULL;

	ocalloc_size += (str != NULL && sgx_is_within_enclave(str, _len_str)) ? _len_str : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_open_file_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_open_file_t));

	if (str != NULL && sgx_is_within_enclave(str, _len_str)) {
		ms->ms_str = (char*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_str);
		memcpy((void*)ms->ms_str, str, _len_str);
	} else if (str == NULL) {
		ms->ms_str = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	status = sgx_ocall(1, ms);


	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL close_file()
{
	sgx_status_t status = SGX_SUCCESS;
	status = sgx_ocall(2, NULL);

	return status;
}

sgx_status_t SGX_CDECL read_file(char* dest, int sz)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_dest = sz;

	ms_read_file_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_read_file_t);
	void *__tmp = NULL;

	ocalloc_size += (dest != NULL && sgx_is_within_enclave(dest, _len_dest)) ? _len_dest : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_read_file_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_read_file_t));

	if (dest != NULL && sgx_is_within_enclave(dest, _len_dest)) {
		ms->ms_dest = (char*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_dest);
		memset(ms->ms_dest, 0, _len_dest);
	} else if (dest == NULL) {
		ms->ms_dest = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	ms->ms_sz = sz;
	status = sgx_ocall(3, ms);

	if (dest) memcpy((void*)dest, ms->ms_dest, _len_dest);

	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL mojo_sleep(unsigned int milliseconds)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_mojo_sleep_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_mojo_sleep_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_mojo_sleep_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_mojo_sleep_t));

	ms->ms_milliseconds = milliseconds;
	status = sgx_ocall(4, ms);


	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL open_networkfile(int* retval, const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_open_networkfile_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_open_networkfile_t);
	void *__tmp = NULL;

	ocalloc_size += (str != NULL && sgx_is_within_enclave(str, _len_str)) ? _len_str : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_open_networkfile_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_open_networkfile_t));

	if (str != NULL && sgx_is_within_enclave(str, _len_str)) {
		ms->ms_str = (char*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_str);
		memcpy((void*)ms->ms_str, str, _len_str);
	} else if (str == NULL) {
		ms->ms_str = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	status = sgx_ocall(5, ms);

	if (retval) *retval = ms->ms_retval;

	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL open_outputnetworkfile(int* retval, const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_open_outputnetworkfile_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_open_outputnetworkfile_t);
	void *__tmp = NULL;

	ocalloc_size += (str != NULL && sgx_is_within_enclave(str, _len_str)) ? _len_str : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_open_outputnetworkfile_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_open_outputnetworkfile_t));

	if (str != NULL && sgx_is_within_enclave(str, _len_str)) {
		ms->ms_str = (char*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_str);
		memcpy((void*)ms->ms_str, str, _len_str);
	} else if (str == NULL) {
		ms->ms_str = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	status = sgx_ocall(6, ms);

	if (retval) *retval = ms->ms_retval;

	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_fprint_networkfile(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_fprint_networkfile_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_fprint_networkfile_t);
	void *__tmp = NULL;

	ocalloc_size += (str != NULL && sgx_is_within_enclave(str, _len_str)) ? _len_str : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_fprint_networkfile_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_fprint_networkfile_t));

	if (str != NULL && sgx_is_within_enclave(str, _len_str)) {
		ms->ms_str = (char*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_str);
		memcpy((void*)ms->ms_str, str, _len_str);
	} else if (str == NULL) {
		ms->ms_str = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	status = sgx_ocall(7, ms);


	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_fread_networkfile(char* retval)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_fread_networkfile_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_fread_networkfile_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_fread_networkfile_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_fread_networkfile_t));

	status = sgx_ocall(8, ms);

	if (retval) *retval = ms->ms_retval;

	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_getint(int* retval)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_getint_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_getint_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_getint_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_getint_t));

	status = sgx_ocall(9, ms);

	if (retval) *retval = ms->ms_retval;

	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_getfloat(float* retval)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_getfloat_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_getfloat_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_getfloat_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_getfloat_t));

	status = sgx_ocall(10, ms);

	if (retval) *retval = ms->ms_retval;

	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_read(char* src, int sz)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_src = sz;

	ms_ocall_read_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_read_t);
	void *__tmp = NULL;

	ocalloc_size += (src != NULL && sgx_is_within_enclave(src, _len_src)) ? _len_src : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_read_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_read_t));

	if (src != NULL && sgx_is_within_enclave(src, _len_src)) {
		ms->ms_src = (char*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_src);
		memset(ms->ms_src, 0, _len_src);
	} else if (src == NULL) {
		ms->ms_src = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	ms->ms_sz = sz;
	status = sgx_ocall(11, ms);

	if (src) memcpy((void*)src, ms->ms_src, _len_src);

	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_read_outenclave(uint64_t src, int sz)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_read_outenclave_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_read_outenclave_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_read_outenclave_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_read_outenclave_t));

	ms->ms_src = src;
	ms->ms_sz = sz;
	status = sgx_ocall(12, ms);


	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_write(char* src, int sz)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_src = sz;

	ms_ocall_write_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_write_t);
	void *__tmp = NULL;

	ocalloc_size += (src != NULL && sgx_is_within_enclave(src, _len_src)) ? _len_src : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_write_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_write_t));

	if (src != NULL && sgx_is_within_enclave(src, _len_src)) {
		ms->ms_src = (char*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_src);
		memcpy(ms->ms_src, src, _len_src);
	} else if (src == NULL) {
		ms->ms_src = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	ms->ms_sz = sz;
	status = sgx_ocall(13, ms);


	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL end_this_line()
{
	sgx_status_t status = SGX_SUCCESS;
	status = sgx_ocall(14, NULL);

	return status;
}

sgx_status_t SGX_CDECL close_networkfile()
{
	sgx_status_t status = SGX_SUCCESS;
	status = sgx_ocall(15, NULL);

	return status;
}

sgx_status_t SGX_CDECL close_outputnetworkfile()
{
	sgx_status_t status = SGX_SUCCESS;
	status = sgx_ocall(16, NULL);

	return status;
}

sgx_status_t SGX_CDECL ocall_newmatrix(uint64_t* retval, uint64_t* px, int* size, int cols, int rows, int chans)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_px = 1 * sizeof(*px);
	size_t _len_size = 1 * sizeof(*size);

	ms_ocall_newmatrix_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_newmatrix_t);
	void *__tmp = NULL;

	ocalloc_size += (px != NULL && sgx_is_within_enclave(px, _len_px)) ? _len_px : 0;
	ocalloc_size += (size != NULL && sgx_is_within_enclave(size, _len_size)) ? _len_size : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_newmatrix_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_newmatrix_t));

	if (px != NULL && sgx_is_within_enclave(px, _len_px)) {
		ms->ms_px = (uint64_t*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_px);
		memset(ms->ms_px, 0, _len_px);
	} else if (px == NULL) {
		ms->ms_px = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	if (size != NULL && sgx_is_within_enclave(size, _len_size)) {
		ms->ms_size = (int*)__tmp;
		__tmp = (void *)((size_t)__tmp + _len_size);
		memset(ms->ms_size, 0, _len_size);
	} else if (size == NULL) {
		ms->ms_size = NULL;
	} else {
		sgx_ocfree();
		return SGX_ERROR_INVALID_PARAMETER;
	}
	
	ms->ms_cols = cols;
	ms->ms_rows = rows;
	ms->ms_chans = chans;
	status = sgx_ocall(17, ms);

	if (retval) *retval = ms->ms_retval;
	if (px) memcpy((void*)px, ms->ms_px, _len_px);
	if (size) memcpy((void*)size, ms->ms_size, _len_size);

	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_fill_uniform(uint64_t pmatrix, float range)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_fill_uniform_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_fill_uniform_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_fill_uniform_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_fill_uniform_t));

	ms->ms_pmatrix = pmatrix;
	ms->ms_range = range;
	status = sgx_ocall(18, ms);


	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_fill_normal(uint64_t pmatrix, float std)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_fill_normal_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_fill_normal_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_fill_normal_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_fill_normal_t));

	ms->ms_pmatrix = pmatrix;
	ms->ms_std = std;
	status = sgx_ocall(19, ms);


	sgx_ocfree();
	return status;
}

