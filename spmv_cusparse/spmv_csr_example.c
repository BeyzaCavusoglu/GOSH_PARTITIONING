/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <zlib.h>

#include <algorithm>
#include <string.h>
#include <time.h>

#include "mmio.h"

float* createRandomArray(int n)
{
    float* array = malloc(n * sizeof(float));

    for (int i = 0; i < n; i++)
    { 
        array[i] = 1;
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

int main(int argc, char *argv[]) {
    // Host problem definition
    
    float     alpha           = 1.0f;
    float     beta            = 0.0f;
    
    char line[200];
    const char* filename = "MATRICES.txt";

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
    matcode[4] = '\0';
    
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

    hY = (float*)malloc(A_num_cols * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(float)) )

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
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, dBuffer) )
    
    printf("%s Time (seconds):", argv[1]);

    
    for (int j = 0; j < 1; j++){
        struct timespec t_start, t_end;
        //clock_gettime(CLOCK_MONOTONIC, &t_start);

        cudaEventRecord(start, 0);

        for(int i=0; i < 1000; i++) {
            CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_MV_ALG_DEFAULT, dBuffer) )
            cudaDeviceSynchronize();
        }

        cudaEventRecord(stop, 0);
        //clock_gettime(CLOCK_MONOTONIC, &t_end);
        cudaEventSynchronize(stop);
        float elapsedTime = 0;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        
        /*double timing_duration = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));
        printf("\t%0.6lf", timing_duration);*/

        printf("\t%0.6f", elapsedTime/1000);
    }
    printf("\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
//printf("%f\n", hY[0]);
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )

    free(values);
    free(rowPtr);
    free(colIndex);
    
    fclose(f);
    return EXIT_SUCCESS;
}
