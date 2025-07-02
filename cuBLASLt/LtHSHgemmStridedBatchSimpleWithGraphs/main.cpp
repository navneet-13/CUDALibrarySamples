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

#include "sample_cublasLt_LtHSHgemmStridedBatchSimple.h"
#include "helpers.h"
#include <iostream>

// Helper macro for CUDA calls inside the lambda
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    // H200 GPU has 143 GB VRAM. Use most of it for workspace, e.g., 130 GB (to leave room for other allocations).
    // 130 GB = 130ULL * 1024 * 1024 * 1024 bytes
    TestBench< __half, __half, float> props(
        CUBLAS_OP_N, CUBLAS_OP_N,
        5120,         // m
        5120,       // n
        5120,         // k
        2.0f,         // alpha
        0.0f,         // beta
        132ULL * 1024 * 1024 * 1024, // workspaceSize (130 GB)
        1             // N (batch count)
    );

    props.run([&props] {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        CHECK_CUDA(cudaStreamCreate(&stream));

        // --- 1. Graph Capture ---
        std::cout << "Capturing CUDA graph..." << std::endl;
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        // CHECK_CUDA(cudaMemcpyAsync(props.Adev, props.A.data(), props.A.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));
        // CHECK_CUDA(cudaMemcpyAsync(props.Bdev, props.B.data(), props.B.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));

        LtHSHgemmStridedBatchSimple(props.ltHandle,
                                    stream,
                                    props.transa,
                                    props.transb,
                                    props.m,
                                    props.n,
                                    props.k,
                                    &props.alpha,
                                    props.Adev,
                                    props.lda,
                                    props.m * props.k,
                                    props.Bdev,
                                    props.ldb,
                                    props.k * props.n,
                                    &props.beta,
                                    props.Cdev,
                                    props.ldc,
                                    props.m * props.n,
                                    props.N,
                                    props.workspace,
                                    props.workspaceSize);

            // CHECK_CUDA(cudaMemcpyAsync(props.C.data(), props.Cdev, props.C.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream));

        // --- 2. End Capture and Instantiate ---
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        std::cout << "Graph captured. Instantiating..." << std::endl;
        CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, 0));
        std::cout << "Graph instantiated." << std::endl;

        // --- 3. Launch Graph ---
        int iterations = 100;
        std::cout << "Launching graph " << iterations << " times..." << std::endl;

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; ++i) {
            CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        std::cout << "Graph execution complete." << std::endl;
        std::cout << "Average execution time over " << iterations << " runs: " << milliseconds / iterations << " ms" << std::endl;

        // --- 4. Cleanup Graph Objects ---
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaGraphExecDestroy(graphExec));
        CHECK_CUDA(cudaGraphDestroy(graph));
        CHECK_CUDA(cudaStreamDestroy(stream));
    });

    

    return 0;
}