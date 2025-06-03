#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h> // For hipBLAS
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm> // For std::min

// 编译文件
// hipcc sourcefile_mlp_forward.cpp -o mlp_forward -lhipblas
// 执行文件
// ./mlp_forward 或者 rocprof --stats -o results.csv ./mlp_forward

#define BATCH 1024
#define I 10
#define H 20
#define O 5

#define TILE_SIZE 16 // Used for custom kernels and new bias/activation kernels

#define HIP_CHECK(cmd) \
    do { \
        hipError_t e = cmd; \
        if (e != hipSuccess) { \
            std::cerr << "Failed: HIP error " << hipGetErrorString(e) << " (" << e << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define HIPBLAS_CHECK(cmd) \
    do { \
        hipblasStatus_t status = cmd; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "Failed: hipBLAS error " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// --- Original Custom Fused Kernels ---
// 优化的矩阵乘法内核，使用共享内存和内存合并访问 (remains for reference, not directly used in MLP if hipBLAS is chosen)
__global__ void matmul_kernel_optimized(const double* A, const double* B, double* C, int M, int N, int K) {
    __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile A
        int tile_row = row;
        int tile_col = tile * TILE_SIZE + threadIdx.x;
        if (tile_row < M && tile_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[tile_row * K + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile B
        tile_row = tile * TILE_SIZE + threadIdx.y;
        tile_col = col;
        if (tile_row < K && tile_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[tile_row * N + tile_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 融合的内核：矩阵乘法 + bias + ReLU
__global__ void matmul_bias_relu_kernel(const double* A, const double* B, const double* bias, 
                                        double* C, int M, int N, int K) {
    __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile A
        int tile_row = row;
        int tile_col = tile * TILE_SIZE + threadIdx.x;
        if (tile_row < M && tile_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[tile_row * K + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile B
        tile_row = tile * TILE_SIZE + threadIdx.y;
        tile_col = col;
        if (tile_row < K && tile_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[tile_row * N + tile_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        // Add bias and apply ReLU in one step
        sum += bias[col];
        C[row * N + col] = fmax(0.0, sum);
    }
}

// 简单的矩阵乘法 + bias内核（用于输出层，无ReLU）
__global__ void matmul_bias_kernel(const double* A, const double* B, const double* bias, 
                                   double* C, int M, int N, int K) {
    __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile A
        int tile_row = row;
        int tile_col = tile * TILE_SIZE + threadIdx.x;
        if (tile_row < M && tile_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[tile_row * K + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile B
        tile_row = tile * TILE_SIZE + threadIdx.y;
        tile_col = col;
        if (tile_row < K && tile_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[tile_row * N + tile_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum + bias[col];
    }
}

// --- New Kernels for Bias and Activation (to be used with hipBLAS) ---
// Note: Experimenting with TILE_SIZE (e.g., 8, 16, 32) for these custom bias/activation
// kernels might yield minor performance differences.
// The current TILE_SIZE is defined globally.
__global__ void add_bias_relu_kernel(const double* mat_in, const double* bias, double* mat_out, int M_rows, int N_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M_rows && col < N_cols) {
        double val = mat_in[row * N_cols + col] + bias[col];
        mat_out[row * N_cols + col] = fmax(0.0, val);
    }
}

__global__ void add_bias_kernel(const double* mat_in, const double* bias, double* mat_out, int M_rows, int N_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M_rows && col < N_cols) {
        mat_out[row * N_cols + col] = mat_in[row * N_cols + col] + bias[col];
    }
}


void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    }
}

/*
 MLP GPU Performance Notes:
 The current implementation uses custom HIP kernels with shared memory tiling and kernel fusion
 (matrix multiplication + bias + activation). These are effective optimization strategies.
 For the given matrix dimensions, the performance is already quite good.

 To potentially achieve further significant speedup in matrix multiplication,
 consider using the hipBLAS library, which provides highly optimized routines
 like hipblasDgemm. This would involve replacing the custom matmul kernels
 with calls to the hipBLAS API.
*/
int main() {
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H_output_custom(BATCH * H); // For custom kernel output
    std::vector<double> h_Y_custom(BATCH * O);      // For custom kernel output
    std::vector<double> h_Y_hipblas(BATCH * O);     // For hipBLAS version output

    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    // Additional buffers for hipBLAS path
    double *d_H_intermediate_hipblas, *d_Y_intermediate_hipblas;


    size_t size_X = BATCH * I * sizeof(double);
    size_t size_W1 = I * H * sizeof(double);
    size_t size_B1 = H * sizeof(double);
    size_t size_H = BATCH * H * sizeof(double); // d_H will be reused for both paths
    size_t size_W2 = H * O * sizeof(double);
    size_t size_B2 = O * sizeof(double);
    size_t size_Y = BATCH * O * sizeof(double); // d_Y will be reused

    // 1. 在设备上分配内存
    HIP_CHECK(hipMalloc(&d_X, size_X));
    HIP_CHECK(hipMalloc(&d_W1, size_W1));
    HIP_CHECK(hipMalloc(&d_B1, size_B1));
    HIP_CHECK(hipMalloc(&d_H, size_H)); // Used by both custom and as output of bias_relu for hipBLAS
    HIP_CHECK(hipMalloc(&d_W2, size_W2));
    HIP_CHECK(hipMalloc(&d_B2, size_B2));
    HIP_CHECK(hipMalloc(&d_Y, size_Y)); // Used by both custom and as output of bias for hipBLAS

    HIP_CHECK(hipMalloc(&d_H_intermediate_hipblas, size_H)); // For X*W1 result before bias/relu
    HIP_CHECK(hipMalloc(&d_Y_intermediate_hipblas, size_Y)); // For H_act*W2 result before bias


    // Create HIP Stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // 2. 将主机数据复制到设备 (Asynchronously)
    HIP_CHECK(hipMemcpyAsync(d_X, h_X.data(), size_X, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_W1, h_W1.data(), size_W1, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_B1, h_B1.data(), size_B1, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_W2, h_W2.data(), size_W2, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_B2, h_B2.data(), size_B2, hipMemcpyHostToDevice, stream));

    // 使用TILE_SIZE定义grid和block维度
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    // --- Performance Measurement for Custom Fused Kernels ---
    hipEvent_t custom_start_event, custom_stop_event, custom_layer1_event, custom_layer2_event;
    HIP_CHECK(hipEventCreate(&custom_start_event));
    HIP_CHECK(hipEventCreate(&custom_stop_event));
    HIP_CHECK(hipEventCreate(&custom_layer1_event));
    HIP_CHECK(hipEventCreate(&custom_layer2_event));

    HIP_CHECK(hipEventRecord(custom_start_event, stream));

    // 隐藏层：使用融合内核 H = ReLU(X * W1 + B1)
    dim3 numBlocks1_custom((H + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(matmul_bias_relu_kernel, numBlocks1_custom, threadsPerBlock, 0, stream,
                       d_X, d_W1, d_B1, d_H, BATCH, H, I);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(custom_layer1_event, stream));

    // 输出层：Y = H * W2 + B2 (无ReLU)
    dim3 numBlocks2_custom((O + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(matmul_bias_kernel, numBlocks2_custom, threadsPerBlock, 0, stream,
                       d_H, d_W2, d_B2, d_Y, BATCH, O, H);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(custom_layer2_event, stream));

    HIP_CHECK(hipMemcpyAsync(h_Y_custom.data(), d_Y, size_Y, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipEventRecord(custom_stop_event, stream));
    HIP_CHECK(hipEventSynchronize(custom_stop_event)); // Wait for all custom kernel operations to complete

    float custom_total_time = 0, custom_layer1_time = 0, custom_layer2_time = 0;
    HIP_CHECK(hipEventElapsedTime(&custom_total_time, custom_start_event, custom_stop_event));
    HIP_CHECK(hipEventElapsedTime(&custom_layer1_time, custom_start_event, custom_layer1_event));
    HIP_CHECK(hipEventElapsedTime(&custom_layer2_time, custom_layer1_event, custom_layer2_event));

    std::cout << "=== Performance Analysis (Custom Fused Kernels) ===" << std::endl;
    std::cout << "GPU MLP (Custom Kernels) Forward Pass Time: " << custom_total_time << " ms" << std::endl;
    std::cout << "Hidden Layer (Custom Fused) Time: " << custom_layer1_time << " ms" << std::endl;
    std::cout << "Output Layer (Custom Fused) Time: " << custom_layer2_time << " ms" << std::endl;
    std::cout << "Memory Transfer Overhead (Custom): " << custom_total_time - custom_layer1_time - custom_layer2_time << " ms" << std::endl;
    
    // --- Performance Measurement for hipBLAS + Custom Bias/Activation Kernels ---
    hipblasHandle_t blas_handle;
    HIPBLAS_CHECK(hipblasCreate(&blas_handle));
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, stream)); // Associate hipBLAS operations with our stream

    hipEvent_t hipblas_start_event, hipblas_stop_event, hipblas_layer1_matmul_event, hipblas_layer1_bias_relu_event, hipblas_layer2_matmul_event, hipblas_layer2_bias_event;
    HIP_CHECK(hipEventCreate(&hipblas_start_event));
    HIP_CHECK(hipEventCreate(&hipblas_stop_event));
    HIP_CHECK(hipEventCreate(&hipblas_layer1_matmul_event));
    HIP_CHECK(hipEventCreate(&hipblas_layer1_bias_relu_event));
    HIP_CHECK(hipEventCreate(&hipblas_layer2_matmul_event));
    HIP_CHECK(hipEventCreate(&hipblas_layer2_bias_event));
    
    // Ensure H2D transfers are complete before starting hipBLAS path timing
    // (though they were on the same stream, an explicit sync before new measurement is good practice)
    HIP_CHECK(hipStreamSynchronize(stream)); 

    HIP_CHECK(hipEventRecord(hipblas_start_event, stream));

    const double alpha = 1.0;
    const double beta = 0.0;

    // Layer 1: H_intermediate = X * W1 (using hipBLAS)
    // hipblasDgemm expects column-major. For row-major C=A*B, call with (B, A) and then C.
    // C(m,n) = A(m,k) * B(k,n) (row-major)
    // hipblasDgemm(handle, opN, opN, n_cols_B, m_rows_A, k_common, alpha, B_ptr, ldb_B_cols, A_ptr, lda_A_cols, beta, C_ptr, ldc_C_cols)
    // Here: d_H_intermediate_hipblas(BATCH,H) = d_X(BATCH,I) * d_W1(I,H)
    // m=BATCH, k=I, n=H
    HIPBLAS_CHECK(hipblasDgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                               H, BATCH, I, // n, m, k
                               &alpha,
                               d_W1, H,      // B_ptr, ldb (cols of W1)
                               d_X, I,       // A_ptr, lda (cols of X)
                               &beta,
                               d_H_intermediate_hipblas, H)); // C_ptr, ldc (cols of H_intermediate)
    HIP_CHECK(hipEventRecord(hipblas_layer1_matmul_event, stream));

    // Layer 1: H_activated = ReLU(H_intermediate + B1) (custom kernel)
    // d_H will store the activated hidden layer output
    dim3 numBlocks_bias_relu((H + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(add_bias_relu_kernel, numBlocks_bias_relu, threadsPerBlock, 0, stream,
                       d_H_intermediate_hipblas, d_B1, d_H, BATCH, H);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(hipblas_layer1_bias_relu_event, stream));
    
    // Layer 2: Y_intermediate = H_activated * W2 (using hipBLAS)
    // d_Y_intermediate_hipblas(BATCH,O) = d_H(BATCH,H) * d_W2(H,O)
    // m=BATCH, k=H, n=O
    HIPBLAS_CHECK(hipblasDgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                               O, BATCH, H, // n, m, k
                               &alpha,
                               d_W2, O,      // B_ptr, ldb (cols of W2)
                               d_H, H,       // A_ptr, lda (cols of H)
                               &beta,
                               d_Y_intermediate_hipblas, O)); // C_ptr, ldc (cols of Y_intermediate)
    HIP_CHECK(hipEventRecord(hipblas_layer2_matmul_event, stream));

    // Layer 2: Y = Y_intermediate + B2 (custom kernel)
    // d_Y will store the final output
    dim3 numBlocks_bias((O + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(add_bias_kernel, numBlocks_bias, threadsPerBlock, 0, stream,
                       d_Y_intermediate_hipblas, d_B2, d_Y, BATCH, O);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(hipblas_layer2_bias_event, stream));

    HIP_CHECK(hipMemcpyAsync(h_Y_hipblas.data(), d_Y, size_Y, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipEventRecord(hipblas_stop_event, stream));
    HIP_CHECK(hipEventSynchronize(hipblas_stop_event)); // Wait for all hipBLAS path operations

    float hipblas_total_time = 0, hipblas_l1_matmul_time = 0, hipblas_l1_bias_relu_time = 0;
    float hipblas_l2_matmul_time = 0, hipblas_l2_bias_time = 0;
    HIP_CHECK(hipEventElapsedTime(&hipblas_total_time, hipblas_start_event, hipblas_stop_event));
    HIP_CHECK(hipEventElapsedTime(&hipblas_l1_matmul_time, hipblas_start_event, hipblas_layer1_matmul_event));
    HIP_CHECK(hipEventElapsedTime(&hipblas_l1_bias_relu_time, hipblas_layer1_matmul_event, hipblas_layer1_bias_relu_event));
    HIP_CHECK(hipEventElapsedTime(&hipblas_l2_matmul_time, hipblas_layer1_bias_relu_event, hipblas_layer2_matmul_event));
    HIP_CHECK(hipEventElapsedTime(&hipblas_l2_bias_time, hipblas_layer2_matmul_event, hipblas_layer2_bias_event));
    
    float hipblas_layer1_combined_time = 0;
    HIP_CHECK(hipEventElapsedTime(&hipblas_layer1_combined_time, hipblas_start_event, hipblas_layer1_bias_relu_event));
    float hipblas_layer2_combined_time = 0;
    HIP_CHECK(hipEventElapsedTime(&hipblas_layer2_combined_time, hipblas_layer1_bias_relu_event, hipblas_layer2_bias_event));


    std::cout << "\n=== Performance Analysis (hipBLAS + Custom Bias/Activation Kernels) ===" << std::endl;
    std::cout << "GPU MLP (hipBLAS path) Forward Pass Time: " << hipblas_total_time << " ms" << std::endl;
    std::cout << "Layer 1 (hipBLAS MatMul) Time: " << hipblas_l1_matmul_time << " ms" << std::endl;
    std::cout << "Layer 1 (Custom Bias/ReLU) Time: " << hipblas_l1_bias_relu_time << " ms" << std::endl;
    std::cout << "Layer 1 Combined (MatMul + Bias/ReLU) Time: " << hipblas_layer1_combined_time << " ms" << std::endl;
    std::cout << "Layer 2 (hipBLAS MatMul) Time: " << hipblas_l2_matmul_time << " ms" << std::endl;
    std::cout << "Layer 2 (Custom Bias) Time: " << hipblas_l2_bias_time << " ms" << std::endl;
    std::cout << "Layer 2 Combined (MatMul + Bias) Time: " << hipblas_layer2_combined_time << " ms" << std::endl;
    std::cout << "Memory Transfer Overhead (hipBLAS path): " << hipblas_total_time - (hipblas_layer1_combined_time + hipblas_layer2_combined_time) << " ms" << std::endl;


    // Print a few output values from both methods to verify consistency (optional)
    std::cout << "\nFirst 5 output vectors (Custom Kernels) (first " << O << " elements each):" << std::endl;
    for (int i = 0; i < std::min(BATCH, 5) ; ++i) {
       std::cout << "Output Batch [" << i << "]: ";
       for (int j = 0; j < O; ++j)
           std::cout << h_Y_custom[i * O + j] << " ";
       std::cout << std::endl;
    }
    std::cout << "First 5 output vectors (hipBLAS Path) (first " << O << " elements each):" << std::endl;
    for (int i = 0; i < std::min(BATCH, 5) ; ++i) {
       std::cout << "Output Batch [" << i << "]: ";
       for (int j = 0; j < O; ++j)
           std::cout << h_Y_hipblas[i * O + j] << " ";
       std::cout << std::endl;
    }


    // Cleanup Events
    HIP_CHECK(hipEventDestroy(custom_start_event));
    HIP_CHECK(hipEventDestroy(custom_stop_event));
    HIP_CHECK(hipEventDestroy(custom_layer1_event));
    HIP_CHECK(hipEventDestroy(custom_layer2_event));

    HIP_CHECK(hipEventDestroy(hipblas_start_event));
    HIP_CHECK(hipEventDestroy(hipblas_stop_event));
    HIP_CHECK(hipEventDestroy(hipblas_layer1_matmul_event));
    HIP_CHECK(hipEventDestroy(hipblas_layer1_bias_relu_event));
    HIP_CHECK(hipEventDestroy(hipblas_layer2_matmul_event));
    HIP_CHECK(hipEventDestroy(hipblas_layer2_bias_event));
    
    // Cleanup hipBLAS
    HIPBLAS_CHECK(hipblasDestroy(blas_handle));

    // Cleanup Stream
    HIP_CHECK(hipStreamDestroy(stream));

    // 6. 释放设备内存
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_W1));
    HIP_CHECK(hipFree(d_B1));
    HIP_CHECK(hipFree(d_H));
    HIP_CHECK(hipFree(d_W2));
    HIP_CHECK(hipFree(d_B2));
    HIP_CHECK(hipFree(d_Y));
    HIP_CHECK(hipFree(d_H_intermediate_hipblas));
    HIP_CHECK(hipFree(d_Y_intermediate_hipblas));

    return 0;
}
