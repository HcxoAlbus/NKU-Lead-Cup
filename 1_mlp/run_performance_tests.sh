#!/bin/bash

# 如果命令以非零状态退出，则立即退出。
set -e

# --- 配置 ---
OUTPUT_DIR="bin" # 输出目录
RESULTS_DIR="results" # 结果目录
MLP_CPP_FILE="sourcefile_mlp_forward.cpp" # MLP前向传播的源文件
MLP_EXEC_NAME="mlp_forward" # MLP前向传播的可执行文件名
CPU_MATMUL_CPP_FILE="../base/sourcefile.cpp" # CPU矩阵乘法的源文件
CPU_MATMUL_EXEC_NAME="cpu_matmul" # CPU矩阵乘法的可执行文件名
DCU_MATMUL_CPP_FILE="../base/sourcefile_dcu.cpp" # DCU矩阵乘法基线的源文件
DCU_MATMUL_EXEC_NAME="dcu_matmul_baseline" # DCU矩阵乘法基线的可执行文件名

PYTHON_VIS_SCRIPT="vis_performance.py" # Python可视化脚本
# 性能测试配置
# 定义测试矩阵大小（DCNN中使用的常见矩阵大小）
MATRIX_SIZES=(128 256 512 1024 2048 4096)
# 神经网络层配置 - 格式: 输入尺寸,隐藏层1,隐藏层2,...,输出尺寸
NN_LAYERS="1024,512,256,10"
# 性能数据文件（将被清理）
MLP_GPU_TIME_FILE="${RESULTS_DIR}/mlp_gpu_time.txt" # MLP GPU时间文件
CPU_PERF_CSV="${RESULTS_DIR}/cpu_performance_data.csv" # CPU性能数据文件
DCU_PERF_CSV="${RESULTS_DIR}/dcu_performance_data.csv" # DCU性能数据文件

# --- 创建目录 ---
mkdir -p ${OUTPUT_DIR} # 创建输出目录
mkdir -p ${RESULTS_DIR} # 创建结果目录

# --- 清理之前的结果 ---
echo "清理之前的结果..."
rm -f ${MLP_GPU_TIME_FILE} ${CPU_PERF_CSV} ${DCU_PERF_CSV} # 删除之前的性能数据文件
# 如果C++程序在当前目录中创建了CSV文件，也一并删除
rm -f cpu_performance_data.csv dcu_performance_data.csv

# --- 编译 ---
echo "编译MLP前向传播（使用HIP和hipBLAS）..."
hipcc -O3 ${MLP_CPP_FILE} -o ${OUTPUT_DIR}/${MLP_EXEC_NAME} -lhipblas

echo "编译CPU矩阵乘法（使用MPI和OpenMP）..."
mpic++ -O3 -fopenmp ${CPU_MATMUL_CPP_FILE} -o ${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME}

echo "编译DCU矩阵乘法基线（使用HIP）..."
hipcc -O3 ${DCU_MATMUL_CPP_FILE} -o ${OUTPUT_DIR}/${DCU_MATMUL_EXEC_NAME}

echo "编译完成。"
echo "----------------------------------------"

# --- 运行MLP前向传播 ---
echo "在DCU上运行MLP前向传播..."
# 捕获完整输出以供Python脚本详细解析
${OUTPUT_DIR}/${MLP_EXEC_NAME} > ${RESULTS_DIR}/mlp_raw_output.txt
# 将原始输出复制到可视化脚本需要的文件
# Python脚本将从该文件中解析必要的行。
cp ${RESULTS_DIR}/mlp_raw_output.txt ${MLP_GPU_TIME_FILE}
# 如果希望在运行时在控制台上查看输出，可以使用tee然后再复制：
# ${OUTPUT_DIR}/${MLP_EXEC_NAME} | tee ${RESULTS_DIR}/mlp_raw_output.txt
# cp ${RESULTS_DIR}/mlp_raw_output.txt ${MLP_GPU_TIME_FILE}

echo "MLP GPU详细输出已保存到${MLP_GPU_TIME_FILE}"
echo "----------------------------------------"

# --- 运行CPU矩阵乘法测试 ---
# 注意：C++程序sourcefile.cpp会将数据追加到执行目录中的'cpu_performance_data.csv'。
# 我们将在所有CPU测试完成后将其移动到RESULTS_DIR。

echo "运行CPU矩阵乘法基线..."
${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} baseline
echo "运行CPU矩阵乘法OpenMP..."
${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} openmp
echo "运行CPU矩阵乘法块划分（块大小=32）..."
${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} block 32
echo "运行CPU矩阵乘法块划分改进版（块大小=32）..."
${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} block_improved 32
# 如果需要，可以添加更多块大小或其他CPU方法

echo "运行CPU矩阵乘法MPI（4个进程）..."
mpirun --allow-run-as-root -np 4 ${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} mpi

# 移动生成的CPU性能数据
if [ -f "cpu_performance_data.csv" ]; then
    mv cpu_performance_data.csv ${CPU_PERF_CSV}
    echo "CPU性能数据已保存到${CPU_PERF_CSV}"
else
    echo "警告：在CPU测试后未找到cpu_performance_data.csv文件。"
fi
echo "----------------------------------------"

# --- 运行DCU矩阵乘法基线测试 ---
# 注意：C++程序sourcefile_dcu.cpp会将数据追加到执行目录中的'dcu_performance_data.csv'。
# 我们将在测试完成后将其移动到RESULTS_DIR。
echo "运行DCU矩阵乘法基线..."
${OUTPUT_DIR}/${DCU_MATMUL_EXEC_NAME}

# 移动生成的DCU性能数据
if [ -f "dcu_performance_data.csv" ]; then
    mv dcu_performance_data.csv ${DCU_PERF_CSV}
    echo "DCU性能数据已保存到${DCU_PERF_CSV}"
else
    echo "警告：在DCU测试后未找到dcu_performance_data.csv文件。"
fi
echo "----------------------------------------"

# --- 理论分析及工具使用提醒 ---
echo "性能分析注意事项："
echo "1. 在DCU程序执行期间，可在单独的终端中使用'rocm-smi'监控利用率。"
echo "   示例：watch -n 1 rocm-smi"
echo "2. 如果需要调试DCU内核，可使用'rocgdb ${OUTPUT_DIR}/${MLP_EXEC_NAME}'（如果适用）。"
echo "----------------------------------------"

# --- 运行Python可视化脚本 ---
echo "运行Python可视化脚本..."
python3 ${PYTHON_VIS_SCRIPT} ${MLP_GPU_TIME_FILE} ${CPU_PERF_CSV} ${DCU_PERF_CSV} ${RESULTS_DIR}
echo "可视化脚本运行完成。图表应已生成在${RESULTS_DIR}"
echo "----------------------------------------"
echo "所有测试已完成。"

# 使脚本可执行：chmod +x run_performance_tests.sh
