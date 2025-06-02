#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>
#include <chrono> // 用于计时
#include <fstream> // 用于文件输出
#include <string>    // 用于 std::to_string
#include <iomanip>   // 用于 std::fixed 和 std::setprecision

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
            C[i * P + j] = 0; // 确保 C 被正确初始化或重置
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
// 使用 collapse(2)
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, int N, int M, int P, int block_size) { // block_size 作为参数
	std::cout << "matmul_block_tiling (collapse(2)) with block_size " << block_size << " methods..." << std::endl;
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

// 方式2的变体: 子块并行，但仅并行化最外层循环
void matmul_block_tiling_outer_omp(const std::vector<double>& A,
                                   const std::vector<double>& B,
                                   std::vector<double>& C, int N, int M, int P, int block_size) {
    std::cout << "matmul_block_tiling_outer_omp (outer loop parallel) with block_size " << block_size << " methods..." << std::endl;
    // C 矩阵应由调用者初始化为0。
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii += block_size) {
        for (int jj = 0; jj < P; jj += block_size) {
            for (int kk = 0; kk < M; kk += block_size) {
                for (int i = ii; i < std::min(ii + block_size, N); ++i) {
                    for (int j = jj; j < std::min(jj + block_size, P); ++j) {
                        for (int k = kk; k < std::min(kk + block_size, M); ++k) {
                            C[i * P + j] += A[i * M + k] * B[k * P + j];
                        }
                    }
                }
            }
        }
    }
}

// 改进的分块算法：减少开销，优化内存访问模式
void matmul_block_tiling_improved(const std::vector<double>& A,
                                  const std::vector<double>& B,
                                  std::vector<double>& C, int N, int M, int P, int block_size) {
    std::cout << "matmul_block_tiling_improved with block_size " << block_size << " methods..." << std::endl;
    
    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < N; ii += block_size) {
        const int i_end = std::min(ii + block_size, N);
        for (int jj = 0; jj < P; jj += block_size) {
            const int j_end = std::min(jj + block_size, P);
            for (int kk = 0; kk < M; kk += block_size) {
                const int k_end = std::min(kk + block_size, M);
                
                // 内层循环展开并优化：使用(i,k,j)顺序改善缓存局部性
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        const double a_ik = A[i * M + k];
                        for (int j = jj; j < j_end; ++j) {
                            C[i * P + j] += a_ik * B[k * P + j];
                        }
                    }
                }
            }
        }
    }
}

// 缓存优化的分块算法：针对特定缓存大小优化
void matmul_block_tiling_cache_opt(const std::vector<double>& A,
                                   const std::vector<double>& B,
                                   std::vector<double>& C, int N, int M, int P, int block_size) {
    std::cout << "matmul_block_tiling_cache_opt with block_size " << block_size << " methods..." << std::endl;
    
    // 使用更细粒度的并行策略
    #pragma omp parallel
    {
        #pragma omp for schedule(guided) nowait
        for (int ii = 0; ii < N; ii += block_size) {
            const int i_end = std::min(ii + block_size, N);
            
            for (int jj = 0; jj < P; jj += block_size) {
                const int j_end = std::min(jj + block_size, P);
                
                for (int kk = 0; kk < M; kk += block_size) {
                    const int k_end = std::min(kk + block_size, M);
                    
                    // 使用寄存器缓存优化和向量化
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            const double a_val = A[i * M + k];
                            // 向量化友好的内层循环
                            #pragma omp simd
                            for (int j = jj; j < j_end; ++j) {
                                C[i * P + j] += a_val * B[k * P + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

// 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
void matmul_mpi(int N, int M, int P, double& elapsed_time) { // 添加 elapsed_time引用
    std::cout << "matmul_mpi methods..." << std::endl;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double mpi_start_time = 0.0, mpi_end_time = 0.0;
    if (rank == 0) {
        mpi_start_time = MPI_Wtime();
    }

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
        mpi_end_time = MPI_Wtime();
        elapsed_time = mpi_end_time - mpi_start_time;
        std::cout << "[MPI] Computation done. Result gathered on rank 0." << std::endl;
        std::cout << "[MPI] Execution Time: " << std::fixed << std::setprecision(6) << elapsed_time << " s" << std::endl;
        
        // 将结果写入 CSV 文件
        std::ofstream outfile("cpu_performance_data.csv", std::ios_base::app);
        if (outfile.is_open()) {
            // 检查文件是否为空，如果为空则写入表头
            outfile.seekp(0, std::ios_base::end); // 转到文件末尾
            if (outfile.tellp() == 0) {
                outfile << "Method,Time\n";
            }
            outfile << "MPI_np" << size << "," << std::fixed << std::setprecision(6) << elapsed_time << "\n";
            outfile.close();
        } else {
            std::cerr << "Unable to open cpu_performance_data.csv for MPI results." << std::endl;
        }
        // 可选：在 rank 0 上进行验证
        // std::vector<double> C_ref_mpi(N * P);
        // init_matrix(A_global, N, M); // 确保 A_global 被初始化以进行验证
        // init_matrix(B_global, M, P); // 确保 B_global 被初始化以进行验证
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
    double elapsed_seconds = 0.0;
    std::string method_name_for_csv;


    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        // elapsed_time 会在 matmul_mpi 内部处理并写入文件
        matmul_mpi(N, M, P, elapsed_seconds); // elapsed_seconds 会被 matmul_mpi 更新 (仅rank 0)
        MPI_Finalize();
        return 0; // MPI 模式在此处返回，因为它自己处理文件写入
    }

    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0); 
    std::vector<double> C_ref(N * P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);
    
    // 首先计算一次基准C_ref，用于验证，这个时间不计入对比，除非mode是baseline
    matmul_baseline(A, B, C_ref, N, M, P);


    if (mode == "baseline") {
        method_name_for_csv = "Baseline";
        auto start = std::chrono::high_resolution_clock::now();
        matmul_baseline(A, B, C, N, M, P); // 重新计算C以计时
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        elapsed_seconds = diff.count();
        std::cout << "[Baseline] Time: " << std::fixed << std::setprecision(6) << elapsed_seconds << " s\n";
        std::cout << "[Baseline] Valid: " << validate(C, C_ref, N, P) << std::endl;
    } else if (mode == "openmp") {
        method_name_for_csv = "OpenMP";
        auto start = std::chrono::high_resolution_clock::now();
        matmul_openmp(A, B, C, N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        elapsed_seconds = diff.count();
        std::cout << "[OpenMP] Time: " << std::fixed << std::setprecision(6) << elapsed_seconds << " s\n";
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P) << std::endl;
    } else if (mode == "block" || mode == "block_outer_omp" || mode == "block_improved" || mode == "block_cache_opt") {
        if (argc < 3) {
            std::cerr << "错误: 模式 " << mode << " 需要一个 block_size 参数。" << std::endl;
            return 1;
        }
        int block_size_val = 0;
        try {
            block_size_val = std::stoi(argv[2]);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "错误: 无效的 block_size 参数: " << argv[2] << std::endl;
            return 1;
        } catch (const std::out_of_range& oor) {
            std::cerr << "错误: block_size 参数超出范围: " << argv[2] << std::endl;
            return 1;
        }

        if (block_size_val <= 0) {
            std::cerr << "错误: block_size 必须为正数。" << std::endl;
            return 1;
        }
        
        std::fill(C.begin(), C.end(), 0.0); // 重置 C 矩阵
        auto start = std::chrono::high_resolution_clock::now();

        if (mode == "block") {
            method_name_for_csv = "BlockTiling_bs" + std::to_string(block_size_val);
            matmul_block_tiling(A, B, C, N, M, P, block_size_val);
        } else if (mode == "block_outer_omp") {
            method_name_for_csv = "BlockTilingOuterOmp_bs" + std::to_string(block_size_val);
            matmul_block_tiling_outer_omp(A, B, C, N, M, P, block_size_val);
        } else if (mode == "block_improved") {
            method_name_for_csv = "BlockTilingImproved_bs" + std::to_string(block_size_val);
            matmul_block_tiling_improved(A, B, C, N, M, P, block_size_val);
        } else { // mode == "block_cache_opt"
            method_name_for_csv = "BlockTilingCacheOpt_bs" + std::to_string(block_size_val);
            matmul_block_tiling_cache_opt(A, B, C, N, M, P, block_size_val);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        elapsed_seconds = diff.count();
        std::cout << "[" << method_name_for_csv << "] Time: " << std::fixed << std::setprecision(6) << elapsed_seconds << " s\n";
        std::cout << "[" << method_name_for_csv << "] Valid: " << validate(C, C_ref, N, P) << std::endl;

    } else if (mode == "other") {
        method_name_for_csv = "OtherLoopOrder";
        // 重置 C 矩阵为 0
        std::fill(C.begin(), C.end(), 0.0);
        auto start = std::chrono::high_resolution_clock::now();
        matmul_other(A, B, C, N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        elapsed_seconds = diff.count();
        std::cout << "[Other] Time: " << std::fixed << std::setprecision(6) << elapsed_seconds << " s\n";
        std::cout << "[Other] Valid: " << validate(C, C_ref, N, P) << std::endl;
    } else {
        std::cerr << "用法: ./outputfile [baseline|openmp|block <block_size>|block_outer_omp <block_size>|block_improved <block_size>|block_cache_opt <block_size>|mpi|other]" << std::endl;
        return 1;
    }

    // 将结果写入 CSV 文件 (非 MPI 模式)
    std::ofstream outfile("cpu_performance_data.csv", std::ios_base::app);
    if (outfile.is_open()) {
        outfile.seekp(0, std::ios_base::end);
        if (outfile.tellp() == 0) {
            outfile << "Method,Time\n";
        }
        outfile << method_name_for_csv << "," << std::fixed << std::setprecision(6) << elapsed_seconds << "\n";
        outfile.close();
    } else {
        std::cerr << "Unable to open cpu_performance_data.csv for " << method_name_for_csv << " results." << std::endl;
    }
    
    return 0;
}
