#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>  // 用于 CPU 计时
#include <fstream> // 用于文件输出
#include <iomanip> // 用于 std::fixed 和 std::setprecision

// 编译
// hipcc sourcefile_dcu.cpp -o outputfile_dcu
// 执行
// ./outputfile_dcu

#define N 1024
#define M 2048 // 修改 M 的值为 2048
#define P 512

// 矩阵乘法的 GPU 核函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int n_rows, int m_cols_A_rows_B, int p_cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 当前线程对应的行
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程对应的列

    if (row < n_rows && col < p_cols_B) { // 确保线程在有效范围内
        double sum = 0.0; // 初始化累加器
        for (int k_idx = 0; k_idx < m_cols_A_rows_B; ++k_idx) { // 遍历中间维度
            sum += A[row * m_cols_A_rows_B + k_idx] * B[k_idx * p_cols_B + col]; // 累加结果
        }
        C[row * p_cols_B + col] = sum; // 将结果写入输出矩阵
    }
}

// 初始化矩阵数据
void init_matrix(std::vector<double>& mat, size_t num_elements) { // 调整签名以匹配用法
    std::mt19937 gen(42); // 使用固定的种子以保证可复现性
    std::uniform_real_distribution<double> dist(-1.0, 1.0); // 原始范围
    for (size_t i = 0; i < num_elements; ++i) // 使用 size_t 迭代
        mat[i] = dist(gen); // 生成随机数并赋值
}

// 使用 CPU 进行矩阵乘法
void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; ++i) // 遍历矩阵 A 的行
        for (int j = 0; j < P; ++j) { // 遍历矩阵 B 的列
            double sum = 0.0; // 初始化累加器
            for (int k = 0; k < M; ++k) // 遍历中间维度
                sum += A[i * M + k] * B[k * P + j]; // 累加结果
            C[i * P + j] = sum; // 将结果写入输出矩阵
        }
    return;
}

// 验证结果是否正确
bool validate(const std::vector<double>& ref, const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i) // 遍历所有元素
        if (std::abs(ref[i] - test[i]) > 1e-6) // 检查误差是否超过阈值
            return false; // 如果有误差，返回 false
    return true; // 如果所有元素都正确，返回 true
}

int main() {
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P); // 分配矩阵内存
    init_matrix(A, N * M); // 初始化矩阵 A
    init_matrix(B, M * P); // 初始化矩阵 B

    // CPU 基准测试计时
    auto cpu_start_time = std::chrono::high_resolution_clock::now(); // 获取开始时间
    matmul_cpu(A, B, C_ref); // 使用 CPU 进行矩阵乘法
    auto cpu_end_time = std::chrono::high_resolution_clock::now(); // 获取结束时间
    std::chrono::duration<double> cpu_duration = cpu_end_time - cpu_start_time; // 计算持续时间
    double cpu_elapsed_seconds = cpu_duration.count(); // 转换为秒
    std::cout << "[CPU Baseline] Time: " << std::fixed << std::setprecision(6) << cpu_elapsed_seconds << " s\n"; // 输出结果

    // GPU 部分
    double *d_A, *d_B, *d_C; // 设备端矩阵指针
    size_t size_A_bytes = (size_t)N * M * sizeof(double); // 矩阵 A 的字节大小
    size_t size_B_bytes = (size_t)M * P * sizeof(double); // 矩阵 B 的字节大小
    size_t size_C_bytes = (size_t)N * P * sizeof(double); // 矩阵 C 的字节大小

    hipError_t err; // 错误变量

    // 分配设备内存
    err = hipMalloc((void**)&d_A, size_A_bytes);
    if (err != hipSuccess) { std::cerr << "hipMalloc d_A failed: " << hipGetErrorString(err) << std::endl; return 1; }
    err = hipMalloc((void**)&d_B, size_B_bytes);
    if (err != hipSuccess) { std::cerr << "hipMalloc d_B failed: " << hipGetErrorString(err) << std::endl; hipFree(d_A); return 1; }
    err = hipMalloc((void**)&d_C, size_C_bytes);
    if (err != hipSuccess) { std::cerr << "hipMalloc d_C failed: " << hipGetErrorString(err) << std::endl; hipFree(d_A); hipFree(d_B); return 1; }
    
    dim3 threadsPerBlock(16, 16); // 每个块的线程数
    dim3 numBlocks((unsigned int)(P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (unsigned int)(N + threadsPerBlock.y - 1) / threadsPerBlock.y); // 块数

    hipEvent_t kernel_start, kernel_stop; // 核函数计时事件
    hipEvent_t total_hip_start, total_hip_stop; // 总 HIP 操作计时事件

    // 创建事件
    err = hipEventCreate(&kernel_start);
    if (err != hipSuccess) { std::cerr << "hipEventCreate kernel_start failed: " << hipGetErrorString(err) << std::endl; return 1; }
    err = hipEventCreate(&kernel_stop);
    if (err != hipSuccess) { std::cerr << "hipEventCreate kernel_stop failed: " << hipGetErrorString(err) << std::endl; hipEventDestroy(kernel_start); return 1; }
    err = hipEventCreate(&total_hip_start);
    if (err != hipSuccess) { std::cerr << "hipEventCreate total_hip_start failed: " << hipGetErrorString(err) << std::endl; hipEventDestroy(kernel_start); hipEventDestroy(kernel_stop); return 1; }
    err = hipEventCreate(&total_hip_stop);
    if (err != hipSuccess) { std::cerr << "hipEventCreate total_hip_stop failed: " << hipGetErrorString(err) << std::endl; hipEventDestroy(kernel_start); hipEventDestroy(kernel_stop); hipEventDestroy(total_hip_start); return 1; }

    // 记录总 HIP 操作开始时间
    err = hipEventRecord(total_hip_start, 0);
    if (err != hipSuccess) { std::cerr << "hipEventRecord total_hip_start failed: " << hipGetErrorString(err) << std::endl; return 1; }

    // 将数据从主机复制到设备
    err = hipMemcpy(d_A, A.data(), size_A_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "hipMemcpy H2D d_A failed: " << hipGetErrorString(err) << std::endl; return 1; }
    err = hipMemcpy(d_B, B.data(), size_B_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "hipMemcpy H2D d_B failed: " << hipGetErrorString(err) << std::endl; hipFree(d_A); return 1; }

    // 记录核函数开始时间
    err = hipEventRecord(kernel_start, 0);
    if (err != hipSuccess) { std::cerr << "hipEventRecord kernel_start failed: " << hipGetErrorString(err) << std::endl; return 1; }
    
    // 启动核函数
    hipLaunchKernelGGL(matmul_kernel, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, N, M, P);
    err = hipGetLastError(); // 检查核函数启动错误
    if (err != hipSuccess) { std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl; return 1; }

    // 记录核函数结束时间
    err = hipEventRecord(kernel_stop, 0);
    if (err != hipSuccess) { std::cerr << "hipEventRecord kernel_stop failed: " << hipGetErrorString(err) << std::endl; return 1; }
    
    // 将结果从设备复制到主机
    err = hipMemcpy(C.data(), d_C, size_C_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "hipMemcpy D2H d_C failed: " << hipGetErrorString(err) << std::endl; return 1; }

    // 记录总 HIP 操作结束时间
    err = hipEventRecord(total_hip_stop, 0);
    if (err != hipSuccess) { std::cerr << "hipEventRecord total_hip_stop failed: " << hipGetErrorString(err) << std::endl; return 1; }
    
    // 同步总 HIP 操作结束事件
    err = hipEventSynchronize(total_hip_stop);
    if (err != hipSuccess) { std::cerr << "hipEventSynchronize total_hip_stop failed: " << hipGetErrorString(err) << std::endl; return 1; }

    // 计算核函数执行时间
    float kernel_milliseconds = 0;
    err = hipEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_stop); 
    if (err != hipSuccess) { std::cerr << "hipEventElapsedTime kernel failed: " << hipGetErrorString(err) << std::endl; return 1; }
    double hip_kernel_seconds = kernel_milliseconds / 1000.0;
    std::cout << "[HIP Kernel] Execution Time: " << std::fixed << std::setprecision(6) << hip_kernel_seconds << " s" << std::endl;

    // 计算总 HIP 操作时间
    float total_hip_milliseconds = 0;
    err = hipEventElapsedTime(&total_hip_milliseconds, total_hip_start, total_hip_stop);
    if (err != hipSuccess) { std::cerr << "hipEventElapsedTime total_hip failed: " << hipGetErrorString(err) << std::endl; return 1; }
    double total_hip_seconds = total_hip_milliseconds / 1000.0;
    std::cout << "[HIP Total (Memcpy+Kernel)] Execution Time: " << std::fixed << std::setprecision(6) << total_hip_seconds << " s" << std::endl;

    // 销毁事件
    hipEventDestroy(kernel_start); 
    hipEventDestroy(kernel_stop);
    hipEventDestroy(total_hip_start);
    hipEventDestroy(total_hip_stop);

    // 验证结果
    if (validate(C_ref, C)) {
       std::cout << "[HIP] Valid: 1" << std::endl;
    } else {
       std::cout << "[HIP] Valid: 0" << std::endl;
    }

    // 释放设备内存
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    // 将结果写入 CSV 文件
    std::ofstream outfile("dcu_performance_data.csv", std::ios_base::app);
    if (outfile.is_open()) {
        outfile.seekp(0, std::ios_base::end);
        if (outfile.tellp() == 0) { // 如果文件为空，写入表头
            outfile << "Method,Time\n";
        }
        outfile << "CPU_Baseline_for_DCU_comparison," << std::fixed << std::setprecision(6) << cpu_elapsed_seconds << "\n";
        outfile << "HIP_Kernel," << std::fixed << std::setprecision(6) << hip_kernel_seconds << "\n";
        outfile << "HIP_Total_Incl_Memcpy," << std::fixed << std::setprecision(6) << total_hip_seconds << "\n";
        outfile.close();
    } else {
        std::cerr << "Unable to open dcu_performance_data.csv for results." << std::endl;
    }
    
    return 0;
}
