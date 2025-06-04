#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h> // 用于hipBLAS
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm> // 用于std::min

// 编译文件
// hipcc sourcefile_mlp_forward.cpp -o mlp_forward -lhipblas
// 执行文件
// ./mlp_forward 或者 rocprof --stats -o results.csv ./mlp_forward

#define BATCH 1024
#define I 10
#define H 20
#define O 5

#define TILE_SIZE 16 // 用于自定义内核和新的偏置/激活内核

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

// --- 原始自定义融合内核 ---
// 优化的矩阵乘法内核，使用共享内存和内存合并访问 (保留作为参考，如果选择hipBLAS则不直接用于MLP)
__global__ void matmul_kernel_optimized(const double* A, const double* B, double* C, int M, int N, int K) {
    __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载瓦片A
        int tile_row = row;
        int tile_col = tile * TILE_SIZE + threadIdx.x;
        if (tile_row < M && tile_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[tile_row * K + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // 加载瓦片B
        tile_row = tile * TILE_SIZE + threadIdx.y;
        tile_col = col;
        if (tile_row < K && tile_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[tile_row * N + tile_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // 计算部分和
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
        // 加载瓦片A
        int tile_row = row;
        int tile_col = tile * TILE_SIZE + threadIdx.x;
        if (tile_row < M && tile_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[tile_row * K + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // 加载瓦片B
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
        // 一步完成添加偏置和应用ReLU
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
        // 加载瓦片A
        int tile_row = row;
        int tile_col = tile * TILE_SIZE + threadIdx.x;
        if (tile_row < M && tile_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[tile_row * K + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // 加载瓦片B
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

// --- 用于偏置和激活的新内核（与hipBLAS一起使用）---
// 注意：对这些自定义偏置/激活内核的TILE_SIZE进行实验（例如8、16、32）
// 可能会产生轻微的性能差异。
// 当前的TILE_SIZE是全局定义的。
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
 MLP GPU 性能说明：
 当前的实现使用带有共享内存分块和内核融合的自定义HIP内核
 （矩阵乘法 + 偏置 + 激活）。这些是有效的优化策略。
 对于给定的矩阵维度，性能已经相当不错。

 为了可能进一步显著提高矩阵乘法的速度，
 考虑使用hipBLAS库，它提供了高度优化的例程，
 例如hipblasDgemm。这将涉及用hipBLAS API调用
 替换自定义的matmul内核。
*/
int main() {
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H_output_custom(BATCH * H); // 用于自定义内核输出
    std::vector<double> h_Y_custom(BATCH * O);      // 用于自定义内核输出
    std::vector<double> h_Y_hipblas(BATCH * O);     // 用于hipBLAS版本输出

    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    // hipBLAS路径的额外缓冲区
    double *d_H_intermediate_hipblas, *d_Y_intermediate_hipblas;


    size_t size_X = BATCH * I * sizeof(double);
    size_t size_W1 = I * H * sizeof(double);
    size_t size_B1 = H * sizeof(double);
    size_t size_H = BATCH * H * sizeof(double); // d_H将同时用于两个路径
    size_t size_W2 = H * O * sizeof(double);
    size_t size_B2 = O * sizeof(double);
    size_t size_Y = BATCH * O * sizeof(double); // d_Y将被重用

    // 1. 在设备上分配内存
    HIP_CHECK(hipMalloc(&d_X, size_X));
    HIP_CHECK(hipMalloc(&d_W1, size_W1));
    HIP_CHECK(hipMalloc(&d_B1, size_B1));
    HIP_CHECK(hipMalloc(&d_H, size_H)); // 用于自定义路径和作为hipBLAS的bias_relu输出
    HIP_CHECK(hipMalloc(&d_W2, size_W2));
    HIP_CHECK(hipMalloc(&d_B2, size_B2));
    HIP_CHECK(hipMalloc(&d_Y, size_Y)); // 用于自定义路径和作为hipBLAS的bias输出

    HIP_CHECK(hipMalloc(&d_H_intermediate_hipblas, size_H)); // 用于X*W1结果，在偏置/relu之前
    HIP_CHECK(hipMalloc(&d_Y_intermediate_hipblas, size_Y)); // 用于H_act*W2结果，在偏置之前


    // 创建HIP流
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // 2. 将主机数据复制到设备 (异步)
    HIP_CHECK(hipMemcpyAsync(d_X, h_X.data(), size_X, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_W1, h_W1.data(), size_W1, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_B1, h_B1.data(), size_B1, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_W2, h_W2.data(), size_W2, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_B2, h_B2.data(), size_B2, hipMemcpyHostToDevice, stream));

    // 使用TILE_SIZE定义grid和block维度
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    // --- 自定义融合内核的性能测量 ---
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
    HIP_CHECK(hipEventSynchronize(custom_stop_event)); // 等待所有自定义内核操作完成

    float custom_total_time = 0, custom_layer1_time = 0, custom_layer2_time = 0;
    HIP_CHECK(hipEventElapsedTime(&custom_total_time, custom_start_event, custom_stop_event));
    HIP_CHECK(hipEventElapsedTime(&custom_layer1_time, custom_start_event, custom_layer1_event));
    HIP_CHECK(hipEventElapsedTime(&custom_layer2_time, custom_layer1_event, custom_layer2_event));

    std::cout << "=== 性能分析 (自定义融合内核) ===" << std::endl;
    std::cout << "GPU MLP (自定义内核) 前向传播时间: " << custom_total_time << " ms" << std::endl;
    std::cout << "隐藏层 (自定义融合) 时间: " << custom_layer1_time << " ms" << std::endl;
    std::cout << "输出层 (自定义融合) 时间: " << custom_layer2_time << " ms" << std::endl;
    std::cout << "内存传输开销 (自定义): " << custom_total_time - custom_layer1_time - custom_layer2_time << " ms" << std::endl;
    
    // --- hipBLAS + 自定义偏置/激活内核的性能测量 ---
    hipblasHandle_t blas_handle;
    HIPBLAS_CHECK(hipblasCreate(&blas_handle));
    HIPBLAS_CHECK(hipblasSetStream(blas_handle, stream)); // 将hipBLAS操作与我们的流关联

    hipEvent_t hipblas_start_event, hipblas_stop_event, hipblas_layer1_matmul_event, hipblas_layer1_bias_relu_event, hipblas_layer2_matmul_event, hipblas_layer2_bias_event;
    HIP_CHECK(hipEventCreate(&hipblas_start_event));
    HIP_CHECK(hipEventCreate(&hipblas_stop_event));
    HIP_CHECK(hipEventCreate(&hipblas_layer1_matmul_event));
    HIP_CHECK(hipEventCreate(&hipblas_layer1_bias_relu_event));
    HIP_CHECK(hipEventCreate(&hipblas_layer2_matmul_event));
    HIP_CHECK(hipEventCreate(&hipblas_layer2_bias_event));
    
    // 确保H2D传输在开始hipBLAS路径计时前完成
    // (虽然它们在同一个流上，但在新的测量之前进行显式同步是良好的做法)
    HIP_CHECK(hipStreamSynchronize(stream)); 

    HIP_CHECK(hipEventRecord(hipblas_start_event, stream));

    const double alpha = 1.0;
    const double beta = 0.0;

    // 第1层: H_intermediate = X * W1 (使用hipBLAS)
    // hipblasDgemm期望列主序。对于行主序C=A*B，调用时使用(B, A)然后C。
    // C(m,n) = A(m,k) * B(k,n) (行主序)
    // hipblasDgemm(handle, opN, opN, n_cols_B, m_rows_A, k_common, alpha, B_ptr, ldb_B_cols, A_ptr, lda_A_cols, beta, C_ptr, ldc_C_cols)
    // 这里: d_H_intermediate_hipblas(BATCH,H) = d_X(BATCH,I) * d_W1(I,H)
    // m=BATCH, k=I, n=H
    HIPBLAS_CHECK(hipblasDgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                               H, BATCH, I, // n, m, k
                               &alpha,
                               d_W1, H,      // B_ptr, ldb (W1的列数)
                               d_X, I,       // A_ptr, lda (X的列数)
                               &beta,
                               d_H_intermediate_hipblas, H)); // C_ptr, ldc (H_intermediate的列数)
    HIP_CHECK(hipEventRecord(hipblas_layer1_matmul_event, stream));

    // 第1层: H_activated = ReLU(H_intermediate + B1) (自定义内核)
    // d_H将存储激活后的隐藏层输出
    dim3 numBlocks_bias_relu((H + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(add_bias_relu_kernel, numBlocks_bias_relu, threadsPerBlock, 0, stream,
                       d_H_intermediate_hipblas, d_B1, d_H, BATCH, H);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(hipblas_layer1_bias_relu_event, stream));
    
    // 第2层: Y_intermediate = H_activated * W2 (使用hipBLAS)
    // d_Y_intermediate_hipblas(BATCH,O) = d_H(BATCH,H) * d_W2(H,O)
    // m=BATCH, k=H, n=O
    HIPBLAS_CHECK(hipblasDgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                               O, BATCH, H, // n, m, k
                               &alpha,
                               d_W2, O,      // B_ptr, ldb (W2的列数)
                               d_H, H,       // A_ptr, lda (H的列数)
                               &beta,
                               d_Y_intermediate_hipblas, O)); // C_ptr, ldc (Y_intermediate的列数)
    HIP_CHECK(hipEventRecord(hipblas_layer2_matmul_event, stream));

    // 第2层: Y = Y_intermediate + B2 (自定义内核)
    // d_Y将存储最终输出
    dim3 numBlocks_bias((O + TILE_SIZE - 1) / TILE_SIZE, (BATCH + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(add_bias_kernel, numBlocks_bias, threadsPerBlock, 0, stream,
                       d_Y_intermediate_hipblas, d_B2, d_Y, BATCH, O);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(hipblas_layer2_bias_event, stream));

    HIP_CHECK(hipMemcpyAsync(h_Y_hipblas.data(), d_Y, size_Y, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipEventRecord(hipblas_stop_event, stream));
    HIP_CHECK(hipEventSynchronize(hipblas_stop_event)); // 等待所有hipBLAS路径操作

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


    std::cout << "\n=== 性能分析 (hipBLAS + 自定义偏置/激活内核) ===" << std::endl;
    std::cout << "GPU MLP (hipBLAS路径) 前向传播时间: " << hipblas_total_time << " ms" << std::endl;
    std::cout << "第1层 (hipBLAS矩阵乘法) 时间: " << hipblas_l1_matmul_time << " ms" << std::endl;
    std::cout << "第1层 (自定义偏置/ReLU) 时间: " << hipblas_l1_bias_relu_time << " ms" << std::endl;
    std::cout << "第1层综合 (矩阵乘法 + 偏置/ReLU) 时间: " << hipblas_layer1_combined_time << " ms" << std::endl;
    std::cout << "第2层 (hipBLAS矩阵乘法) 时间: " << hipblas_l2_matmul_time << " ms" << std::endl;
    std::cout << "第2层 (自定义偏置) 时间: " << hipblas_l2_bias_time << " ms" << std::endl;
    std::cout << "第2层综合 (矩阵乘法 + 偏置) 时间: " << hipblas_layer2_combined_time << " ms" << std::endl;
    std::cout << "内存传输开销 (hipBLAS路径): " << hipblas_total_time - (hipblas_layer1_combined_time + hipblas_layer2_combined_time) << " ms" << std::endl;


    // 打印两种方法的几个输出值以验证一致性（可选）
    std::cout << "\n前5个输出向量 (自定义内核) (每个的前" << O << "个元素):" << std::endl;
    for (int i = 0; i < std::min(BATCH, 5) ; ++i) {
       std::cout << "输出批次 [" << i << "]: ";
       for (int j = 0; j < O; ++j)
           std::cout << h_Y_custom[i * O + j] << " ";
       std::cout << std::endl;
    }
    std::cout << "前5个输出向量 (hipBLAS路径) (每个的前" << O << "个元素):" << std::endl;
    for (int i = 0; i < std::min(BATCH, 5) ; ++i) {
       std::cout << "输出批次 [" << i << "]: ";
       for (int j = 0; j < O; ++j)
           std::cout << h_Y_hipblas[i * O + j] << " ";
       std::cout << std::endl;
    }


    // 清理事件
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
    
    // 清理hipBLAS
    HIPBLAS_CHECK(hipblasDestroy(blas_handle));

    // 清理流
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
