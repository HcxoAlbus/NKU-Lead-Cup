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

// 主要修改函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int n_rows, int m_cols_A_rows_B, int p_cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_rows && col < p_cols_B) {
        double sum = 0.0;
        for (int k_idx = 0; k_idx < m_cols_A_rows_B; ++k_idx) {
            sum += A[row * m_cols_A_rows_B + k_idx] * B[k_idx * p_cols_B + col];
        }
        C[row * p_cols_B + col] = sum;
    }
}

void init_matrix(std::vector<double>& mat, size_t num_elements) { // 调整签名以匹配用法
    std::mt19937 gen(42); // 使用固定的种子以保证可复现性
    std::uniform_real_distribution<double> dist(-1.0, 1.0); // 原始范围
    for (size_t i = 0; i < num_elements; ++i) // 使用 size_t 迭代
        mat[i] = dist(gen);
}

void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
    return;
}

bool validate(const std::vector<double>& ref, const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::abs(ref[i] - test[i]) > 1e-6)
            return false;
    return true;
}

int main() {
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A, N * M); 
    init_matrix(B, M * P); 

    // CPU baseline timing
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_ref);
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end_time - cpu_start_time;
    double cpu_elapsed_seconds = cpu_duration.count();
    std::cout << "[CPU Baseline] Time: " << std::fixed << std::setprecision(6) << cpu_elapsed_seconds << " s\n";

    // 主要修改部分
    // Allocate and copy to device, use matmul_kernel to compute in DCU
    double *d_A, *d_B, *d_C;
    size_t size_A_bytes = (size_t)N * M * sizeof(double);
    size_t size_B_bytes = (size_t)M * P * sizeof(double);
    size_t size_C_bytes = (size_t)N * P * sizeof(double);

    hipError_t err;

    // 性能分析提示:
    // 1. 使用 rocprof:
    //    例如: rocprof --stats -o dcu_results.csv ./outputfile_dcu
    //    rocprof 会记录 HIP API 调用和内核执行时间。
    // 2. 使用 HIP 事件进行手动计时:
    //    hipEvent_t start_event, stop_event;
    //    hipEventCreate(&start_event);
    //    hipEventCreate(&stop_event);
    //
    //    // 记录数据传输和内核执行总时间
    //    hipEventRecord(start_event, 0);
    //
    //    hipMemcpy(d_A, ...);
    //    hipMemcpy(d_B, ...);
    //    hipLaunchKernelGGL(matmul_kernel, ...);
    //    hipMemcpy(C.data(), ...); // D2H 拷贝也是阻塞的，会等待内核完成
    //
    //    hipEventRecord(stop_event, 0);
    //    hipEventSynchronize(stop_event); // 确保事件已完成
    //    float milliseconds = 0;
    //    hipEventElapsedTime(&milliseconds, start_event, stop_event);
    //    std::cout << "[HIP Total] Time: " << milliseconds / 1000.0 << " s\n";
    //
    //    // 只记录内核执行时间
    //    // hipEventRecord(kernel_start_event, 0);
    //    // hipLaunchKernelGGL(matmul_kernel, ...);
    //    // hipEventRecord(kernel_stop_event, 0);
    //    // hipEventSynchronize(kernel_stop_event);
    //    // hipEventElapsedTime(&kernel_milliseconds, kernel_start_event, kernel_stop_event);
    //    // std::cout << "[HIP Kernel] Time: " << kernel_milliseconds / 1000.0 << " s\n";
    //
    //    // hipEventDestroy(start_event);
    //    // hipEventDestroy(stop_event);
    //    // ... (destroy kernel events too)
    // 将这些时间输出或记录到文件，然后用于 Python 可视化脚本。


    err = hipMalloc((void**)&d_A, size_A_bytes);
    if (err != hipSuccess) { std::cerr << "hipMalloc d_A failed: " << hipGetErrorString(err) << std::endl; return 1; }
    err = hipMalloc((void**)&d_B, size_B_bytes);
    if (err != hipSuccess) { std::cerr << "hipMalloc d_B failed: " << hipGetErrorString(err) << std::endl; hipFree(d_A); return 1; }
    err = hipMalloc((void**)&d_C, size_C_bytes);
    if (err != hipSuccess) { std::cerr << "hipMalloc d_C failed: " << hipGetErrorString(err) << std::endl; hipFree(d_A); hipFree(d_B); return 1; }
	
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((unsigned int)(P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (unsigned int)(N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    hipEvent_t kernel_start, kernel_stop;
    hipEvent_t total_hip_start, total_hip_stop; // Declare total_hip_start and total_hip_stop

    err = hipEventCreate(&kernel_start);
    if (err != hipSuccess) { std::cerr << "hipEventCreate kernel_start failed: " << hipGetErrorString(err) << std::endl; /* cleanup */ return 1; }
    err = hipEventCreate(&kernel_stop);
    if (err != hipSuccess) { std::cerr << "hipEventCreate kernel_stop failed: " << hipGetErrorString(err) << std::endl; hipEventDestroy(kernel_start); /* cleanup */ return 1; }
    err = hipEventCreate(&total_hip_start); // For total HIP operation time
    if (err != hipSuccess) { std::cerr << "hipEventCreate total_hip_start failed: " << hipGetErrorString(err) << std::endl; hipEventDestroy(kernel_start); hipEventDestroy(kernel_stop); /* cleanup */ return 1; }
    err = hipEventCreate(&total_hip_stop);
    if (err != hipSuccess) { std::cerr << "hipEventCreate total_hip_stop failed: " << hipGetErrorString(err) << std::endl; hipEventDestroy(kernel_start); hipEventDestroy(kernel_stop); hipEventDestroy(total_hip_start); /* cleanup */ return 1; }

    err = hipEventRecord(total_hip_start, 0); // Start timing total HIP operations
    if (err != hipSuccess) { std::cerr << "hipEventRecord total_hip_start failed: " << hipGetErrorString(err) << std::endl; /* cleanup */ return 1; }

    err = hipMemcpy(d_A, A.data(), size_A_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "hipMemcpy H2D d_A failed: " << hipGetErrorString(err) << std::endl; /* ... free memory ... */ return 1; }
    err = hipMemcpy(d_B, B.data(), size_B_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "hipMemcpy H2D d_B failed: " << hipGetErrorString(err) << std::endl; /* ... free memory ... */ hipFree(d_A); return 1; }


    err = hipEventRecord(kernel_start, 0);
    if (err != hipSuccess) { std::cerr << "hipEventRecord kernel_start failed: " << hipGetErrorString(err) << std::endl; /* cleanup */ return 1; }
    
    hipLaunchKernelGGL(matmul_kernel, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, N, M, P);
    err = hipGetLastError(); // Check kernel launch error separately
    if (err != hipSuccess) { std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl; /* ... free memory and events ... */ return 1; }

    err = hipEventRecord(kernel_stop, 0);
    if (err != hipSuccess) { std::cerr << "hipEventRecord kernel_stop failed: " << hipGetErrorString(err) << std::endl; /* cleanup */ return 1; }
    
    err = hipMemcpy(C.data(), d_C, size_C_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "hipMemcpy D2H d_C failed: " << hipGetErrorString(err) << std::endl; /* ... free memory and events ... */ return 1; }

    err = hipEventRecord(total_hip_stop, 0); // Stop timing total HIP operations
    if (err != hipSuccess) { std::cerr << "hipEventRecord total_hip_stop failed: " << hipGetErrorString(err) << std::endl; /* cleanup */ return 1; }
    
    err = hipEventSynchronize(total_hip_stop); // Ensure all HIP operations are done
    if (err != hipSuccess) { std::cerr << "hipEventSynchronize total_hip_stop failed: " << hipGetErrorString(err) << std::endl; /* cleanup */ return 1; }


    float kernel_milliseconds = 0;
    err = hipEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_stop); 
    if (err != hipSuccess) { std::cerr << "hipEventElapsedTime kernel failed: " << hipGetErrorString(err) << std::endl; /* cleanup */ return 1; }
    double hip_kernel_seconds = kernel_milliseconds / 1000.0;
    std::cout << "[HIP Kernel] Execution Time: " << std::fixed << std::setprecision(6) << hip_kernel_seconds << " s" << std::endl;

    float total_hip_milliseconds = 0;
    err = hipEventElapsedTime(&total_hip_milliseconds, total_hip_start, total_hip_stop);
    if (err != hipSuccess) { std::cerr << "hipEventElapsedTime total_hip failed: " << hipGetErrorString(err) << std::endl; /* cleanup */ return 1; }
    double total_hip_seconds = total_hip_milliseconds / 1000.0;
    std::cout << "[HIP Total (Memcpy+Kernel)] Execution Time: " << std::fixed << std::setprecision(6) << total_hip_seconds << " s" << std::endl;


    // hipEventDestroy(kernel_start); // These will be handled below
    // hipEventDestroy(kernel_stop);
    // hipEventDestroy(total_hip_start);
    // hipEventDestroy(total_hip_stop);

    // Cleanup all events and memory
    // Errors during destroy are typically logged but not critical for program flow.
    hipEventDestroy(kernel_start); 
    hipEventDestroy(kernel_stop);
    hipEventDestroy(total_hip_start);
    hipEventDestroy(total_hip_stop);

    if (validate(C_ref, C)) {
       std::cout << "[HIP] Valid: 1" << std::endl;
    } else {
       std::cout << "[HIP] Valid: 0" << std::endl;
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    // 将结果写入 CSV 文件
    std::ofstream outfile("dcu_performance_data.csv", std::ios_base::app);
    if (outfile.is_open()) {
        outfile.seekp(0, std::ios_base::end);
        if (outfile.tellp() == 0) {
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
