/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cublasLt.h>
#include <iostream>

#include "sample_cublasLt_LtHSHgemmStridedBatchSimple.h"
#include "helpers.h"

/// Sample wrapper executing mixed precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasGemmEx,
/// with addition of the workspace to support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed
void LtHSHgemmStridedBatchSimple(cublasLtHandle_t ltHandle,
                                 cudaStream_t stream,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 size_t m,
                                 size_t n,
                                 size_t k,
                                 const float *alpha, /* host pointer */
                                 const __half *A,
                                 int lda,
                                 int64_t stridea,
                                 const __half *B,
                                 int ldb,
                                 int64_t strideb,
                                 const float *beta, /* host pointer */
                                 __half *C,
                                 int ldc,
                                 int64_t stridec,
                                 int batchCount,
                                 void *workspace,
                                 size_t workspaceSize) {

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we need to configure batch size and counts in this case
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));

    // in this simplified example we take advantage of cublasLtMatmul shortcut notation with algo=NULL which will force
    // matmul to get the basic heuristic result internally. Downsides of this approach are that there is no way to
    // configure search preferences (e.g. disallow tensor operations or some reduction schemes) and no way to store the
    // algo for later use

    cublasLtMatmulPreference_t preference = NULL;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

    // Set max workspace size allowed
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
        preference,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &workspaceSize, sizeof(workspaceSize)));

    const int requestAlgoCount = 10;
    cublasLtMatmulHeuristicResult_t heuristicResults[requestAlgoCount];
    int returnedAlgoCount = 0;

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        operationDesc,
        Adesc,
        Bdesc,
        Cdesc,
        Cdesc,
        preference,
        requestAlgoCount,
        heuristicResults,
        &returnedAlgoCount));

    if (returnedAlgoCount == 0) {
        std::cerr << "No suitable algorithm found.\n";
        return;
    }

    // Choose best one (usually first one)
    auto bestAlgo = &heuristicResults[0].algo;

    

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    // // cublasSetStream(ltHandle, stream);

    // cudaEventRecord(start, stream);

    // cudaEventRecord(start,0);

    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     beta,
                                     C,
                                     Cdesc,
                                      C,
                                     Cdesc,
                                     bestAlgo,
                                     workspace,
                                     workspaceSize,
                                     stream));

// cudaEventRecord(stop,0);
    // cudaEventRecord(stop, stream);
    // cudaEventSynchronize(stop);
    // cudaStreamDestroy(stream);
    // cudaEventRecord(stop,0);
    // cudaEventSynchronize(stop);

    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
