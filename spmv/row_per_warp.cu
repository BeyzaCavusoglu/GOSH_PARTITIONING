#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <zlib.h>

#include <string.h>
#include <time.h>
#include "mmio.c"
#include "smsh.c"

#define THREADS_PER_BLK 32

float* createRandomArray(int n)
{
    float* array = (float*)malloc(n * sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < n; i++)
    {
        float randomValue = (float)rand() / RAND_MAX; 
        array[i] = randomValue * 2 - 1;
    }

    return array;
}

/********************************************************************************************************************************************/
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


__global__
void spmv_1D(int* rowPtr, int* colPtr, float* valPtr, float* dense_vec, float* results, int n_rows) {

    int N = rowPtr[blockIdx.x + 1] - rowPtr[blockIdx.x];

    __shared__ float support[THREADS_PER_BLK+2];

    support[threadIdx.x] = 0.0f;
    __syncthreads();

    /* Exclude out of bounds threads */
    if (threadIdx.x < N) {

        /* Compute multiplication of row*dense_vector */
        for (int i = rowPtr[blockIdx.x] + threadIdx.x; i < rowPtr[blockIdx.x + 1]; i += blockDim.x) {
            support[threadIdx.x] += valPtr[i] * dense_vec[colPtr[i]];
        }
    
        __syncthreads();

        /* Add results together */
        for (int i = blockDim.x / 2; i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                support[threadIdx.x] += support[threadIdx.x + i];
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        results[blockIdx.x] = support[0];
    }
}


int main(int argc, char *argv[]) {
    // Host problem definition

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int A_num_rows, A_num_cols, nz, A_nnz;
    int i = 0, *I_complete, *J_complete;
    float *V_complete;

    /*******************************************************************/
    if ((f = fopen(argv[1], "r")) == NULL)
    {
        printf("Could not locate the matrix file. Please make sure the pathname is valid.\n");
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &A_num_rows, &A_num_cols, &nz)) != 0)
    {
        printf("Could not read matrix dimensions.\n");
        exit(1);
    }
    
    if ((strcmp(matcode, "MCRG") == 0) || (strcmp(matcode, "MCIG") == 0) || (strcmp(matcode, "MCPG") == 0) || (strcmp(matcode, "MCCG") == 0))
    {

        I_complete = (int *)calloc(nz, sizeof(int));
        J_complete = (int *)calloc(nz, sizeof(int));
        V_complete = (float *)calloc(nz, sizeof(float));

        for (i = 0; i < nz; i++)
        {
            if (matcode[2] == 'P') {
                fscanf(f, "%d %d", &I_complete[i], &J_complete[i]);
                V_complete[i] = 1;
            }  
            else {
                fscanf(f, "%d %d %f", &I_complete[i], &J_complete[i], &V_complete[i]);
            } 
            fscanf(f, "%*[^\n]\n");
            /* adjust from 1-based to 0-based */
            I_complete[i]--;
            J_complete[i]--;
        }
    }

    /* If the matrix is symmetric, we need to construct the other half */

    else if ((strcmp(matcode, "MCRS") == 0) || (strcmp(matcode, "MCIS") == 0) || (strcmp(matcode, "MCPS") == 0) || (strcmp(matcode, "MCCS") == 0) || (strcmp(matcode, "MCCH") == 0) || (strcmp(matcode, "MCRK") == 0) || (matcode[0] == 'M' && matcode[1] == 'C' && matcode[2] == 'P' && matcode[3] == 'S'))
    {

        I_complete = (int *)calloc(2 * nz, sizeof(int));
        J_complete = (int *)calloc(2 * nz, sizeof(int));
        V_complete = (float *)calloc(2 * nz, sizeof(float));

        int i_index = 0;

        for (i = 0; i < nz; i++)
        {
            if (matcode[2] == 'P') {
                fscanf(f, "%d %d", &I_complete[i], &J_complete[i]);
                V_complete[i] = 1;
            }
            else {
                fscanf(f, "%d %d %f", &I_complete[i], &J_complete[i], &V_complete[i]);
            }
                
            fscanf(f, "%*[^\n]\n");

            if (I_complete[i] == J_complete[i])
            {
                /* adjust from 1-based to 0-based */
                I_complete[i]--;
                J_complete[i]--;
            }
            else
            {
                /* adjust from 1-based to 0-based */
                I_complete[i]--;
                J_complete[i]--;
                J_complete[nz + i_index] = I_complete[i];
                I_complete[nz + i_index] = J_complete[i];
                V_complete[nz + i_index] = V_complete[i];
                i_index++;
            }
        }
        nz += i_index;
    }
    else
    {
        printf("This matrix type is not supported: %s \n", matcode);
        exit(1);
    }

    /* sort COO array by the rows */
    if (!isSorted(J_complete, I_complete, nz)) {
        quicksort(J_complete, I_complete, V_complete, nz);
    }
    
    /* Convert from COO to CSR */
    int *rowPtr = (int *)calloc(A_num_rows + 1, sizeof(int));
    int *colIndex = (int *)calloc(nz, sizeof(int));
    float *values = (float *)calloc(nz, sizeof(float));

    for (i = 0; i < nz; i++) {
        colIndex[i] = J_complete[i];
        values[i] = V_complete[i];
        rowPtr[I_complete[i] + 1]++;
    }
    for (i = 0; i < A_num_rows + 1; i++) {
        rowPtr[i + 1] += rowPtr[i];
    }
    A_nnz = nz;
    /*******************************************************************/

    float* hY = NULL;
    float* hX = createRandomArray(A_num_cols);

    /* Allocate memory for the vector of the results */
    hY = (float*)malloc(A_num_cols * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;

    /* Allocate memory for the SPMV+CSR vectors */
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(float)) )

    /* Copy SPMV+CSR vectors to memory */
    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, rowPtr,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, colIndex, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(float),
                           cudaMemcpyHostToDevice) )

    // execute SpMV
    dim3 threadsPerBlock(THREADS_PER_BLK, 1, 1);
    dim3 numBlocks(A_num_rows, 1, 1);

    spmv_1D<<<numBlocks,threadsPerBlock>>>(dA_csrOffsets, dA_columns, dA_values, dX, dY, A_num_rows);
	cudaDeviceSynchronize();

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for(int i=0; i < 1000; i++) {
    	
        spmv_1D<<<numBlocks,threadsPerBlock>>>(dA_csrOffsets, dA_columns, dA_values, dX, dY, A_num_rows);
	    cudaDeviceSynchronize();
    }
	
    cudaDeviceSynchronize();

	clock_gettime(CLOCK_MONOTONIC, &t_end);
	double timing_duration = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));
	printf("%s time (seconds):\t%0.6lf\n",argv[1], timing_duration);

    //CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float), cudaMemcpyDeviceToHost) )

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

/*     // device memory deallocation
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )

    free(values);
    free(rowPtr);
    free(colIndex);s
     */
    fclose(f);
    return EXIT_SUCCESS;
}
