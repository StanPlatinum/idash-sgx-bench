//
//  test.cpp
//  
//
//  Created by Sourgroup on 10/31/18.
//

#include <stdio.h>

#include "Enclave_u.h"
#include "sgx_urts.h"

/* Global EID shared by multiple threads */
sgx_enclave_id_t eid = 0;

// OCall implementations
void ocall_print(const char* str) {
    printf("%s\n", str);
}

int main(int argc, char *argv[])
{
    sgx_status_t        ret = SGX_SUCCESS;
	sgx_launch_token_t  token = { 0 };
	int updated = 0;

	ret = sgx_create_enclave("enclave.signed.so", SGX_DEBUG_FLAG, &token, &updated, &eid, NULL);
	if (ret != SGX_SUCCESS)
		return -1;
	// Initializing the enclave finished.	
	generate_random_number(eid);
	sgx_destroy_enclave(eid);
}

