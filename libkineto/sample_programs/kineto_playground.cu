/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if 0
#include <stdio.h>

#include "kineto_playground.cuh"

namespace kineto {

void warmup(void) {
  // Inititalizing CUDA can take a while which we normally do not want to see in
  // Kineto traces. This is done in various ways that take Kineto as dependency.
  // This is our way of doing warmup for kineto_playground
  size_t bytes = 1000;
  float* mem = NULL;
  auto error = cudaMalloc(&mem, bytes);
  if (error != cudaSuccess) {
    printf(
        "cudaMalloc failed during kineto_playground warmup. error code: %d",
        error);
    return;
  }

  cudaFree(mem);
}

float *hA, *dA, *hOut;
int num = 50'000;

void basicMemcpyToDevice(void) {
  size_t size = num * sizeof(float);
  cudaError_t err;

  hA = (float*)malloc(size);
  hOut = (float*)malloc(size);
  err = cudaMalloc(&dA, size);
  if (err != cudaSuccess) {
    printf("cudaMalloc failed during %s", __func__);
    return;
  }

  memset(hA, 1, size);
  err = cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("cudaMemcpy failed during %s", __func__);
    return;
  }
}

void basicMemcpyFromDevice(void) {
  size_t size = num * sizeof(float);
  cudaError_t err;

  err = cudaMemcpy(hOut, dA, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy failed during %s", __func__);
    return;
  }

  free(hA);
  free(hOut);
  cudaFree(dA);
}

__global__ void square(float* A, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    A[i] *= A[i];
  }
}

void playground(void) {
  // Add your experimental CUDA implementation here.
  basicMemcpyFromDevice();
  compute();
  basicMemcpyFromDevice();
}

void compute(void) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;
  for (int i = 0; i < 10; i++) {
    square<<<blocksPerGrid, threadsPerBlock>>>(dA, num);
  }
}

} // namespace kineto
#endif

#include <algorithm>      // For std::min
#include <cuda_runtime.h> // For CUDA API calls
#include <stdio.h> // Keep for fprintf in checkCudaError, but remove printf calls
#include <stdlib.h> // For malloc, free, rand
#include <string.h> // For memset
#include <time.h>   // For time (to seed rand)

// Assuming kineto_playground.cuh declares these functions
// For demonstration, these are defined directly in this .cu file.
namespace kineto {

// --- Configuration Parameters ---
// Size for matrix multiplication (MM_SIZE x MM_SIZE)
const int MM_SIZE = 2048;
// Length for vector operations (10^7 elements)
const int VECTOR_LENGTH = 10000000;
// Number of times to repeat the entire complex workload
const int NUM_ITERATIONS = 10;

// Global pointers for host and device memory for the complex workload
float *h_mm_a, *h_mm_b, *h_mm_c;
float *d_mm_a, *d_mm_b, *d_mm_c;

float *h_vec1, *h_vec2, *h_vec3;
float *d_vec1, *d_vec2, *d_vec3;
// d_result_vec will be used for both the input slice and the element-wise
// output
float *d_result_vec;

float *h_sum_result;
float *d_sum_result; // Device memory for a single float sum result

// Original variables from the provided code, kept separate for clarity
float *hA_orig, *dA_orig, *hOut_orig;
int num_orig = 50000; // Original num for the 'square' kernel

// --- Helper for CUDA Error Checking ---
// This function checks for CUDA errors and prints a message, then exits.
void checkCudaError(cudaError_t err, const char *func_name) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error in %s: %s\n", func_name,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// --- Warmup Function ---
// Performs a small CUDA operation to initialize the CUDA driver and runtime,
// preventing this setup time from appearing in later traces.
void warmup(void) {
  size_t bytes = 1000;
  float *mem = NULL;
  auto error = cudaMalloc(&mem, bytes);
  if (error != cudaSuccess) {
    // fprintf(stderr, "cudaMalloc failed during kineto_playground warmup. error
    // code: %d\n", error);
    return;
  }
  cudaFree(mem);
  // No printf here
}

// --- New Kernels for Complex Workload ---

// Kernel for Matrix Multiplication: C = A * B
// Assumes A, B, and C are square matrices of size N x N.
// Each thread computes one element of the resulting matrix C.
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int N) {
  // Calculate global row and column indices for the current thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    // Perform dot product of row 'row' from A and column 'col' from B
    for (int i = 0; i < N; ++i) {
      sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

// Kernel for Element-wise Operation with dependency:
// ResultVec[i] = Vec1[i] + Vec2[i] * Vec3[i] + SliceFromMatrixC[i]
// SliceFromMatrixC is a 1D array (already prepared on host and copied to
// device).
__global__ void elementWiseOpKernel(float *Vec1, float *Vec2, float *Vec3,
                                    float *SliceFromMatrixC, float *ResultVec,
                                    int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    ResultVec[i] = Vec1[i] + Vec2[i] * Vec3[i] + SliceFromMatrixC[i];
  }
}

// Kernel for Reduction (Sum of all elements in 'input' array)
// Uses shared memory for per-block reduction and atomicAdd for global sum.
__global__ void reduceSumKernel(float *input, float *output, int N) {
  // Shared memory for per-block reduction
  __shared__ float
      sdata[256]; // Assuming blockDim.x <= 256 for this simple example

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data from global memory to shared memory
  sdata[tid] = (i < N) ? input[i] : 0.0f;
  __syncthreads(); // Ensure all threads in block have loaded their data

  // Perform parallel reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads(); // Synchronize after each reduction step
  }

  // The first thread in the block writes the block's sum to global memory
  if (tid == 0) {
    // Use atomicAdd to safely add to the global sum, especially if multiple
    // blocks are contributing to the same 'output' location.
    atomicAdd(output, sdata[0]);
  }
}

// --- Original 'square' Kernel ---
// Multiplies each element of array A by itself.
__global__ void square(float *A, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    A[i] *= A[i];
  }
}

// --- Memory Management Functions ---

// Allocates and initializes all host and device memory required for the
// workload.
void allocateAndInitializeMemory() {
  // No printf here
  // Seed random number generator for consistent data across runs (optional)
  srand(time(NULL));

  size_t mm_size_bytes = MM_SIZE * MM_SIZE * sizeof(float);
  size_t vec_size_bytes = VECTOR_LENGTH * sizeof(float);
  size_t orig_size_bytes = num_orig * sizeof(float);

  // Host memory allocation
  h_mm_a = (float *)malloc(mm_size_bytes);
  h_mm_b = (float *)malloc(mm_size_bytes);
  h_mm_c = (float *)malloc(mm_size_bytes);
  h_vec1 = (float *)malloc(vec_size_bytes);
  h_vec2 = (float *)malloc(vec_size_bytes);
  h_vec3 = (float *)malloc(vec_size_bytes);
  h_sum_result = (float *)malloc(sizeof(float)); // For single sum result

  hA_orig = (float *)malloc(orig_size_bytes);
  hOut_orig = (float *)malloc(orig_size_bytes);

  // Initialize host data with random values for complex workload
  for (int i = 0; i < MM_SIZE * MM_SIZE; ++i) {
    h_mm_a[i] = (float)rand() / RAND_MAX;
    h_mm_b[i] = (float)rand() / RAND_MAX;
  }
  for (int i = 0; i < VECTOR_LENGTH; ++i) {
    h_vec1[i] = (float)rand() / RAND_MAX;
    h_vec2[i] = (float)rand() / RAND_MAX;
    h_vec3[i] = (float)rand() / RAND_MAX;
  }
  // Initialize original data (used by 'square' kernel)
  for (int i = 0; i < num_orig; ++i) {
    hA_orig[i] = 1.0f; // Original code used memset(hA, 1, size) which sets
                       // bytes to 1, not floats.
  }

  // Device memory allocation
  checkCudaError(cudaMalloc(&d_mm_a, mm_size_bytes), "cudaMalloc d_mm_a");
  checkCudaError(cudaMalloc(&d_mm_b, mm_size_bytes), "cudaMalloc d_mm_b");
  checkCudaError(cudaMalloc(&d_mm_c, mm_size_bytes), "cudaMalloc d_mm_c");

  checkCudaError(cudaMalloc(&d_vec1, vec_size_bytes), "cudaMalloc d_vec1");
  checkCudaError(cudaMalloc(&d_vec2, vec_size_bytes), "cudaMalloc d_vec2");
  checkCudaError(cudaMalloc(&d_vec3, vec_size_bytes), "cudaMalloc d_vec3");
  checkCudaError(cudaMalloc(&d_result_vec, vec_size_bytes),
                 "cudaMalloc d_result_vec");

  checkCudaError(cudaMalloc(&d_sum_result, sizeof(float)),
                 "cudaMalloc d_sum_result");

  checkCudaError(cudaMalloc(&dA_orig, orig_size_bytes), "cudaMalloc dA_orig");

  // No printf here
}

// Frees all allocated host and device memory.
void freeAllMemory() {
  // No printf here
  // Free host memory
  free(h_mm_a);
  free(h_mm_b);
  free(h_mm_c);
  free(h_vec1);
  free(h_vec2);
  free(h_vec3);
  free(h_sum_result);
  free(hA_orig);
  free(hOut_orig);

  // Free device memory
  checkCudaError(cudaFree(d_mm_a), "cudaFree d_mm_a");
  checkCudaError(cudaFree(d_mm_b), "cudaFree d_mm_b");
  checkCudaError(cudaFree(d_mm_c), "cudaFree d_mm_c");
  checkCudaError(cudaFree(d_vec1), "cudaFree d_vec1");
  checkCudaError(cudaFree(d_vec2), "cudaFree d_vec2");
  checkCudaError(cudaFree(d_vec3), "cudaFree d_vec3");
  checkCudaError(cudaFree(d_result_vec), "cudaFree d_result_vec");
  checkCudaError(cudaFree(d_sum_result), "cudaFree d_sum_result");
  checkCudaError(cudaFree(dA_orig), "cudaFree dA_orig");
  // No printf here
}

// --- Main Compute Function ---
// This function orchestrates the complex GPU workload.
void compute(void) {
  // No printf here

  // Define grid and block dimensions for matrix multiplication kernel
  dim3 threadsPerBlock_mm(16, 16); // 16x16 threads per block (256 threads)
  dim3 blocksPerGrid_mm(
      (MM_SIZE + threadsPerBlock_mm.x - 1) / threadsPerBlock_mm.x,
      (MM_SIZE + threadsPerBlock_mm.y - 1) / threadsPerBlock_mm.y);

  // Define grid and block dimensions for vector operations and reduction
  int threadsPerBlock_vec = 256;
  int blocksPerGrid_vec =
      (VECTOR_LENGTH + threadsPerBlock_vec - 1) / threadsPerBlock_vec;

  // Define grid and block dimensions for the original 'square' kernel
  int threadsPerBlock_orig = 256;
  int blocksPerGrid_orig =
      (num_orig + threadsPerBlock_orig - 1) / threadsPerBlock_orig;

  // Loop through multiple iterations of the complex workload
  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    // No printf here

    // --- 1. Data Transfer (Host to Device) for this iteration ---
    // Copy input data for matrix multiplication from host to device
    checkCudaError(cudaMemcpy(d_mm_a, h_mm_a, MM_SIZE * MM_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy h_mm_a to d_mm_a");
    checkCudaError(cudaMemcpy(d_mm_b, h_mm_b, MM_SIZE * MM_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy h_mm_b to d_mm_b");
    // Copy input data for vector operations from host to device
    checkCudaError(cudaMemcpy(d_vec1, h_vec1, VECTOR_LENGTH * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy h_vec1 to d_vec1");
    checkCudaError(cudaMemcpy(d_vec2, h_vec2, VECTOR_LENGTH * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy h_vec2 to d_vec2");
    checkCudaError(cudaMemcpy(d_vec3, h_vec3, VECTOR_LENGTH * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy h_vec3 to d_vec3");
    // Copy data for the original 'square' kernel from host to device
    checkCudaError(cudaMemcpy(dA_orig, hA_orig, num_orig * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy hA_orig to dA_orig");

    // Reset the device sum result to zero before each reduction operation
    checkCudaError(cudaMemset(d_sum_result, 0, sizeof(float)),
                   "cudaMemset d_sum_result");
    // Synchronize to ensure all data transfers are complete before kernel
    // launches
    checkCudaError(cudaDeviceSynchronize(),
                   "cudaDeviceSynchronize after initial data transfers");
    // No printf here

    // --- 2. Kernel 1: Matrix Multiplication (MM_SIZE x MM_SIZE) ---
    // No printf here
    matrixMultiplyKernel<<<blocksPerGrid_mm, threadsPerBlock_mm>>>(
        d_mm_a, d_mm_b, d_mm_c, MM_SIZE);
    // Check for asynchronous errors from the kernel launch
    checkCudaError(cudaGetLastError(), "matrixMultiplyKernel launch");
    // Synchronize to ensure matrix multiplication is complete
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after MM");
    // No printf here

    // --- 3. Prepare slice from MM result for Element-wise operation ---
    // This simulates a dependency where part of one kernel's output is used by
    // another. We copy a slice from d_mm_c to a temporary host buffer, pad it
    // if necessary, and then copy it back to d_result_vec on the device.
    int slice_length =
        std::min(VECTOR_LENGTH,
                 MM_SIZE * MM_SIZE); // Ensure slice doesn't exceed matrix size
    float *h_mm_c_slice = (float *)malloc(VECTOR_LENGTH * sizeof(float));
    if (h_mm_c_slice == NULL) {
      fprintf(stderr, "Failed to allocate h_mm_c_slice\n");
      exit(EXIT_FAILURE);
    }

    // Copy the relevant part of d_mm_c (the first 'slice_length' elements) to
    // host
    checkCudaError(cudaMemcpy(h_mm_c_slice, d_mm_c,
                              slice_length * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy d_mm_c slice to host");

    // Pad the host slice with zeros if it's shorter than VECTOR_LENGTH
    if (slice_length < VECTOR_LENGTH) {
      memset(h_mm_c_slice + slice_length, 0,
             (VECTOR_LENGTH - slice_length) * sizeof(float));
    }

    // Copy the prepared (and potentially padded) slice from host to
    // d_result_vec on device
    checkCudaError(cudaMemcpy(d_result_vec, h_mm_c_slice,
                              VECTOR_LENGTH * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy padded slice to d_result_vec");
    free(h_mm_c_slice); // Free the temporary host buffer
    checkCudaError(cudaDeviceSynchronize(),
                   "cudaDeviceSynchronize after slice preparation");
    // No printf here

    // --- 4. Kernel 2: Element-wise Operation (VECTOR_LENGTH) ---
    // No printf here
    elementWiseOpKernel<<<blocksPerGrid_vec, threadsPerBlock_vec>>>(
        d_vec1, d_vec2, d_vec3, d_result_vec, d_result_vec, VECTOR_LENGTH);
    checkCudaError(cudaGetLastError(), "elementWiseOpKernel launch");
    // Synchronize to ensure element-wise operation is complete
    checkCudaError(cudaDeviceSynchronize(),
                   "cudaDeviceSynchronize after Element-wise Op");
    // No printf here

    // --- 5. Kernel 3: Reduction Operation (Sum of d_result_vec) ---
    // No printf here
    reduceSumKernel<<<blocksPerGrid_vec, threadsPerBlock_vec>>>(
        d_result_vec, d_sum_result, VECTOR_LENGTH);
    checkCudaError(cudaGetLastError(), "reduceSumKernel launch");
    // Synchronize to ensure reduction is complete
    checkCudaError(cudaDeviceSynchronize(),
                   "cudaDeviceSynchronize after Reduction");
    // No printf here

    // --- 6. Original 'square' kernel (integrated for more complexity) ---
    // This section runs the original 'square' kernel multiple times.
    // No printf here
    for (int j = 0; j < 10; ++j) { // Original loop of 10 launches
      square<<<blocksPerGrid_orig, threadsPerBlock_orig>>>(dA_orig, num_orig);
      checkCudaError(cudaGetLastError(), "square kernel launch");
    }
    // Synchronize after all launches of the square kernel
    checkCudaError(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize after Original Square kernel operations");
    // No printf here

    // --- 7. Final Data Transfer (Device to Host) ---
    // Copy the final sum result from device to host
    checkCudaError(cudaMemcpy(h_sum_result, d_sum_result, sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy d_sum_result to h_sum_result");
    // Copy the result of the original 'square' kernel from device to host
    checkCudaError(cudaMemcpy(hOut_orig, dA_orig, num_orig * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy dA_orig to hOut_orig");
    // Synchronize to ensure all final data transfers are complete
    checkCudaError(cudaDeviceSynchronize(),
                   "cudaDeviceSynchronize after final data transfers");
    // Removed printf for final sum result
    // No printf here
  }
  // No printf here
}

// --- Playground Function ---
// This is the main entry point for the Kineto playground simulation.
void playground(void) {
  // No printf here
  // Allocate and initialize all necessary memory once at the beginning
  allocateAndInitializeMemory();
  // Run the complex GPU workload
  compute();
  // Free all allocated memory at the end
  freeAllMemory();
  // No printf here
}

} // namespace kineto
