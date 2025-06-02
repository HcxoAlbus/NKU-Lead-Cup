#!/bin/bash
# filepath: d:\sec_semester_code\CPP\Lead_cup\base\run_performance_tests.sh

# --- 配置 ---
CPU_SRC="sourcefile.cpp"
CPU_EXE="outputfile"
DCU_SRC="sourcefile_dcu.cpp"
DCU_EXE="outputfile_dcu"

CPU_PERF_CSV="cpu_performance_data.csv"
DCU_PERF_CSV="dcu_performance_data.csv"
PYTHON_VISUALIZER="visualize_performance.py"

# ROCPROF 输出目录
ROCPROF_CPU_DIR="rocprof_cpu_out"
ROCPROF_DCU_DIR="rocprof_dcu_out"

# MPI 进程数测试列表
MPI_PROCESSES=(1 2 4) # 您可以根据需要修改，例如 (1 2 4 8)

# --- 辅助函数 ---
execute_and_log() {
    echo "Executing: $@"
    "$@"
    if [ $? -ne 0 ]; then
        echo "Error during execution: $@" >&2
        # exit 1 # 可以选择在错误时退出
    fi
}

# --- 主逻辑 ---

echo "===== Performance Test Script ====="

# 0. 设置工作目录 (假设脚本与源文件在同一目录)
BASE_DIR=$(dirname "$0")
cd "$BASE_DIR" || { echo "错误: 无法进入目录 $BASE_DIR"; exit 1; }
echo "当前工作目录: $(pwd)"


# 1. 清理旧的性能数据文件和 rocprof 输出
echo "\n--- 清理旧数据 ---"
rm -f "$CPU_PERF_CSV" "$DCU_PERF_CSV"
rm -rf "$ROCPROF_CPU_DIR" "$ROCPROF_DCU_DIR"
mkdir -p "$ROCPROF_CPU_DIR" "$ROCPROF_DCU_DIR"
echo "旧的 CSV 数据文件已删除。"
echo "旧的 rocprof 输出目录已删除并重新创建。"

# 2. 编译源文件
echo "\n--- 编译代码 ---"
echo "编译 CPU 版本 ($CPU_SRC)..."
execute_and_log mpic++ -fopenmp -o "$CPU_EXE" "$CPU_SRC" -std=c++17
if [ ! -f "$CPU_EXE" ]; then echo "错误: CPU 程序编译失败。"; exit 1; fi

echo "编译 DCU/HIP 版本 ($DCU_SRC)..."
execute_and_log hipcc -o "$DCU_EXE" "$DCU_SRC" -std=c++17
if [ ! -f "$DCU_EXE" ]; then echo "错误: DCU/HIP 程序编译失败。"; exit 1; fi
echo "编译完成。"

# 3. 运行 CPU 基准测试 (由 C++ 程序内部计时并写入 CSV)
echo "\n--- 运行 CPU 基准测试 ---"
execute_and_log "./$CPU_EXE" baseline
execute_and_log "./$CPU_EXE" openmp

# 为 BlockTiling (collapse(2)) 测试不同的块大小
BLOCK_SIZES=(16 32 64 96 128 192 256) # 扩展块大小范围
echo "\n运行 BlockTiling (collapse(2)) 测试..."
for bs in "${BLOCK_SIZES[@]}"; do
    echo "运行 BlockTiling (collapse(2)) 使用 block_size $bs..."
    execute_and_log "./$CPU_EXE" block "$bs"
done

# 为 BlockTilingOuterOmp (仅外层 omp for) 测试不同的块大小
echo "\n运行 BlockTilingOuterOmp (outer omp for) 测试..."
for bs in "${BLOCK_SIZES[@]}"; do
    echo "运行 BlockTilingOuterOmp 使用 block_size $bs..."
    execute_and_log "./$CPU_EXE" block_outer_omp "$bs"
done

# 新增：运行改进的分块版本
echo "\n运行改进的分块优化版本..."
for bs in "${BLOCK_SIZES[@]}"; do
    echo "运行 BlockTilingImproved 使用 block_size $bs..."
    execute_and_log "./$CPU_EXE" block_improved "$bs"
done

# 新增：运行缓存优化分块版本
echo "\n运行缓存优化分块版本..."
for bs in "${BLOCK_SIZES[@]}"; do
    echo "运行 BlockTilingCacheOpt 使用 block_size $bs..."
    execute_and_log "./$CPU_EXE" block_cache_opt "$bs"
done

execute_and_log "./$CPU_EXE" other

# 使用 rocprof 对 OpenMP 版本进行分析 (示例)
# rocprof 主要用于 GPU 分析，但也可以用于 CPU 上的 HSA/ROCr 调用（如果适用）
# 对于纯 CPU OpenMP，其提供的 CPU 指标可能不如 'perf' 等工具详细
# 这里我们主要依赖 C++ 内的计时，但 rocprof 可以提供一些系统级信息
echo "\n运行 rocprof 分析 OpenMP 版本 (主要用于演示)..."
execute_and_log rocprof --stats -o "${ROCPROF_CPU_DIR}/rocprof_openmp_stats.csv" "./$CPU_EXE" openmp
echo "rocprof OpenMP 分析数据（如果成功）保存在 ${ROCPROF_CPU_DIR}/rocprof_openmp_stats.csv"


# 4. 运行 MPI 基准测试 (由 C++ 程序内部计时并写入 CSV)
echo "\n--- 运行 MPI 基准测试 ---"
for np in "${MPI_PROCESSES[@]}"; do
    echo "运行 MPI 使用 $np 个进程..."
    execute_and_log mpirun -np "$np" "./$CPU_EXE" mpi
    
    # MPI 的 rocprof 分析通常更复杂，需要针对每个 rank 或特定 rank
    # 这里仅演示对整个 mpirun 命令的 rocprof 包装，其效用可能有限
    # echo "运行 rocprof 分析 MPI ($np 进程)..."
    # execute_and_log rocprof --stats -o "${ROCPROF_CPU_DIR}/rocprof_mpi_${np}p_stats.csv" mpirun -np "$np" "./$CPU_EXE" mpi
done

# 5. 运行 DCU/HIP 基准测试
echo "\n--- 运行 DCU/HIP 基准测试 ---"
echo "运行 DCU/HIP 程序 (内部计时将写入 $DCU_PERF_CSV)..."
execute_and_log "./$DCU_EXE"

echo "\n使用 rocprof 分析 DCU/HIP 程序..."
# --hip-trace 用于跟踪 HIP API 调用
# --hsa-trace 用于跟踪 HSA API 调用
# --roctx-trace 用于 ROCTX 用户标记 (如果代码中使用了 ROCTX)
# --stats 提供汇总统计
# 您可以根据需要添加更多 rocprof 选项，例如 --metrics ALL_ aktywności GPU
execute_and_log rocprof --timestamp on --hip-trace --hsa-trace --stats -o "${ROCPROF_DCU_DIR}/rocprof_hip_detailed.csv" "./$DCU_EXE"
echo "rocprof DCU/HIP 分析数据保存在 ${ROCPROF_DCU_DIR}/rocprof_hip_detailed.csv"
echo "您可以使用 'rocprof --view <path_to_rocprof_output.db>' (如果生成了 .db 文件) 或查看 .csv 文件获取详细信息。"

# 提示: 使用 rocm-smi 进行实时监控
echo "\n提示: 您可以在运行 GPU 密集型任务时，在另一个终端中使用 'rocm-smi' 或 'rocm-smi -l 1' 进行实时监控。"

# 提示: 使用 rocgdb 进行调试
echo "\n提示: 如果程序崩溃或行为异常，可以使用 rocgdb 进行调试:"
echo "  rocgdb --args ./$CPU_EXE <mode>"
echo "  rocgdb --args ./$DCU_EXE"


# 6. 运行 Python 可视化脚本
echo "\n--- 生成性能图表 ---"
if [ -f "$PYTHON_VISUALIZER" ]; then
    execute_and_log python3 "$PYTHON_VISUALIZER"
    echo "可视化脚本已执行。如果 matplotlib 配置正确，图表应该已显示或保存。"
else
    echo "警告: Python 可视化脚本 '$PYTHON_VISUALIZER' 未找到。"
fi

echo "\n===== 性能测试完成 ====="
echo "CPU 性能数据: $CPU_PERF_CSV"
echo "DCU/HIP 性能数据: $DCU_PERF_CSV"
echo "rocprof CPU 输出目录: $ROCPROF_CPU_DIR"
echo "rocprof DCU/HIP 输出目录: $ROCPROF_DCU_DIR"
