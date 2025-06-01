#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

// 编译执行方式参考：
// 编译， 也可以使用g++，但使用MPI时需使用mpic
// mpic++ -fopenmp -o outputfile sourcefile.cpp

// 运行 baseline
// ./outputfile baseline

// 运行 OpenMP
// ./outputfile openmp

// 运行 子块并行优化
// ./outputfile block

// 运行 MPI（假设 4 个进程）
// mpirun -np 4 ./outputfile mpi

// 运行 其他优化
// ./outputfile other


// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42); // 使用固定的种子以保证可复现性
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证计算优化后的矩阵计算和baseline实现是否结果一致，可以设计其他验证方法，来验证计算的正确性和性能
bool validate(const std::vector<double>& C_computed, const std::vector<double>& C_ref, int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(C_computed[i] - C_ref[i]) > tol) {
            std::cerr << "Validation failed at index " << i << ": computed=" << C_computed[i] << ", ref=" << C_ref[i] << std::endl;
            return false;
        }
    return true;
}

// 基础的矩阵乘法baseline实现（使用一维数组）
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C, int N, int M, int P) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0;
            for (int k = 0; k < M; ++k)
                C[i * P + j] += A[i * M + k] * B[k * P + j];
        }
}

// 方式1: 利用OpenMP进行多线程并发的编程 （主要修改函数）
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C, int N, int M, int P) {
	std::cout << "matmul_openmp methods..." << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// 方式2: 利用子块并行思想，进行缓存友好型的并行优化方法 （主要修改函数)
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, int N, int M, int P, int block_size = 64) {
	std::cout << "matmul_block_tiling methods..." << std::endl;
    // C 矩阵应由调用者初始化为0，main函数中已执行此操作。
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += block_size) {
        for (int jj = 0; jj < P; jj += block_size) {
            for (int kk = 0; kk < M; kk += block_size) {
                for (int i = ii; i < std::min(ii + block_size, N); ++i) {
                    for (int j = jj; j < std::min(jj + block_size, P); ++j) {
                        // C[i*P+j] 在此累加来自不同(kk)块的贡献
                        // double local_sum_for_cij_block = 0.0; // 如果C未初始化或需要在块内累加
                        for (int k = kk; k < std::min(kk + block_size, M); ++k) {
                            C[i * P + j] += A[i * M + k] * B[k * P + j];
                        }
                        // C[i*P+j] += local_sum_for_cij_block; // 如果使用临时块内累加器
                    }
                }
            }
        }
    }
}

// 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
void matmul_mpi(int N, int M, int P) {
    std::cout << "matmul_mpi methods..." << std::endl;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> A_global, B_global(M * P), C_global;
    if (rank == 0) {
        A_global.resize(N * M);
        C_global.resize(N * P);
        init_matrix(A_global, N, M);
        init_matrix(B_global, M, P); // B_global 在 rank 0 初始化
    }

    // 广播 B 矩阵给所有进程
    MPI_Bcast(B_global.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算每个进程处理的行数和偏移量
    int rows_per_process = N / size;
    int remaining_rows = N % size;
    int local_N = (rank < remaining_rows) ? (rows_per_process + 1) : rows_per_process;
    int offset = rank * rows_per_process + std::min(rank, remaining_rows);

    std::vector<double> A_local(local_N * M);
    std::vector<double> C_local(local_N * P, 0.0);

    // 使用 Scatterv 分发 A 矩阵的行
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    if (rank == 0) {
        int current_displ = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = ((i < remaining_rows) ? (rows_per_process + 1) : rows_per_process) * M;
            displs[i] = current_displ;
            current_displ += sendcounts[i];
        }
    }
    MPI_Scatterv(A_global.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), local_N * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // 每个进程计算其负责的 C 矩阵部分
    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A_local[i * M + k] * B_global[k * P + j];
            }
            C_local[i * P + j] = sum;
        }
    }

    // 使用 Gatherv 收集 C 矩阵的各个部分到 rank 0
    std::vector<int> recvcounts(size);
    std::vector<int> recv_displs(size);
    if (rank == 0) {
        int current_recv_displ = 0;
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = ((i < remaining_rows) ? (rows_per_process + 1) : rows_per_process) * P;
            recv_displs[i] = current_recv_displ;
            current_recv_displ += recvcounts[i];
        }
    }
    MPI_Gatherv(C_local.data(), local_N * P, MPI_DOUBLE,
                C_global.data(), recvcounts.data(), recv_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[MPI] Computation done. Result gathered on rank 0." << std::endl;
        // 可选：在 rank 0 上进行验证
        // std::vector<double> C_ref_mpi(N * P);
        // matmul_baseline(A_global, B_global, C_ref_mpi, N, M, P);
        // std::cout << "[MPI] Valid: " << validate(C_global, C_ref_mpi, N, P) << std::endl;
    }
}

// 方式4: 其他方式 （主要修改函数） - 例如：循环顺序优化 (i, k, j)
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C, int N, int M, int P) {
    std::cout << "Other methods (i, k, j loop order)..." << std::endl;
    // C 矩阵应由调用者初始化为0，main函数中已执行此操作。
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < M; ++k) {
            // double val_A_ik = A[i * M + k]; // 可以将A[i*M+k]提出一层循环以减少重复寻址，但现代编译器通常能优化
            for (int j = 0; j < P; ++j) {
                // if (k == 0) C[i * P + j] = 0; // 如果C未被外部初始化为0
                C[i * P + j] += A[i * M + k] * B[k * P + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    // 性能分析提示:
    // 1. 使用 rocprof:
    //    例如: rocprof --stats -o results.csv ./outputfile openmp
    //    然后，您可以解析 results.csv 文件以获取内核执行时间。
    // 2. 在代码中手动计时 (适用于 CPU 端代码):
    //    #include <chrono>
    //    auto start_time = std::chrono::high_resolution_clock::now();
    //    // ... 要计时的代码 ...
    //    auto end_time = std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<double> diff = end_time - start_time;
    //    std::cout << "Execution time: " << diff.count() << " s\n";
    //    将这些时间输出或记录到文件，然后用于 Python 可视化脚本。

    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        // 对于 MPI，通常在 rank 0 记录总时间，或使用 MPI_Wtime()
        // double mpi_start_time, mpi_end_time;
        // if (rank == 0) mpi_start_time = MPI_Wtime();
        matmul_mpi(N, M, P);
        // if (rank == 0) {
        //     mpi_end_time = MPI_Wtime();
        //     std::cout << "MPI Execution time: " << (mpi_end_time - mpi_start_time) << " s\n";
        // }
        MPI_Finalize();
        return 0;
    }

    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0); // C 初始化为0
    std::vector<double> C_ref(N * P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);

    // 计时 baseline
    // auto baseline_start = std::chrono::high_resolution_clock::now();
    matmul_baseline(A, B, C_ref, N, M, P);
    // auto baseline_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> baseline_duration = baseline_end - baseline_start;
    // std::cout << "[Baseline] Time: " << baseline_duration.count() << " s\n";


    if (mode == "baseline") {
        std::cout << "[Baseline] Done.\n";
        // 输出 baseline_duration.count()
    } else if (mode == "openmp") {
        // auto openmp_start = std::chrono::high_resolution_clock::now();
        matmul_openmp(A, B, C, N, M, P);
        // auto openmp_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> openmp_duration = openmp_end - openmp_start;
        // std::cout << "[OpenMP] Time: " << openmp_duration.count() << " s\n";
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P) << std::endl;
        // 输出 openmp_duration.count()
    } else if (mode == "block") {
        // auto block_start = std::chrono::high_resolution_clock::now();
        matmul_block_tiling(A, B, C, N, M, P);
        // auto block_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> block_duration = block_end - block_start;
        // std::cout << "[Block Parallel] Time: " << block_duration.count() << " s\n";
        std::cout << "[Block Parallel] Valid: " << validate(C, C_ref, N, P) << std::endl;
        // 输出 block_duration.count()
    } else if (mode == "other") {
        // auto other_start = std::chrono::high_resolution_clock::now();
        matmul_other(A, B, C, N, M, P);
        // auto other_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> other_duration = other_end - other_start;
        // std::cout << "[Other] Time: " << other_duration.count() << " s\n";
        std::cout << "[Other] Valid: " << validate(C, C_ref, N, P) << std::endl;
        // 输出 other_duration.count()
    } else {
        std::cerr << "Usage: ./main [baseline|openmp|block|mpi|other]" << std::endl; // Added other to usage
    }
	// 需额外增加性能评测代码或其他工具进行评测
    // 例如，将上述计时结果写入一个CSV文件，供 Python 脚本读取
    // std::ofstream outfile("performance_results.csv");
    // outfile << "Method,Time\n";
    // outfile << "Baseline," << baseline_duration.count() << "\n";
    // if (mode == "openmp") outfile << "OpenMP," << openmp_duration.count() << "\n";
    // // ... etc. for other modes
    // outfile.close();
    return 0;
}
