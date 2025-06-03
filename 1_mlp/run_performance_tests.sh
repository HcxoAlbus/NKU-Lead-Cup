#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
OUTPUT_DIR="bin"
RESULTS_DIR="results"
MLP_CPP_FILE="sourcefile_mlp_forward.cpp"
MLP_EXEC_NAME="mlp_forward"
CPU_MATMUL_CPP_FILE="../base/sourcefile.cpp"
CPU_MATMUL_EXEC_NAME="cpu_matmul"
DCU_MATMUL_CPP_FILE="../base/sourcefile_dcu.cpp"
DCU_MATMUL_EXEC_NAME="dcu_matmul_baseline"

PYTHON_VIS_SCRIPT="vis_performance.py"

# Performance data files (will be cleaned)
MLP_GPU_TIME_FILE="${RESULTS_DIR}/mlp_gpu_time.txt"
CPU_PERF_CSV="${RESULTS_DIR}/cpu_performance_data.csv"
DCU_PERF_CSV="${RESULTS_DIR}/dcu_performance_data.csv"

# --- Create directories ---
mkdir -p ${OUTPUT_DIR}
mkdir -p ${RESULTS_DIR}

# --- Clean previous results ---
echo "Cleaning previous results..."
rm -f ${MLP_GPU_TIME_FILE} ${CPU_PERF_CSV} ${DCU_PERF_CSV}
# Also remove CSVs if they are created in the current directory by C++ programs
rm -f cpu_performance_data.csv dcu_performance_data.csv

# --- Compilation ---
echo "Compiling MLP Forward (HIP with hipBLAS)..."
hipcc -O3 ${MLP_CPP_FILE} -o ${OUTPUT_DIR}/${MLP_EXEC_NAME} -lhipblas

echo "Compiling CPU Matmul (MPI & OpenMP)..."
mpic++ -O3 -fopenmp ${CPU_MATMUL_CPP_FILE} -o ${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME}

echo "Compiling DCU Matmul Baseline (HIP)..."
hipcc -O3 ${DCU_MATMUL_CPP_FILE} -o ${OUTPUT_DIR}/${DCU_MATMUL_EXEC_NAME}

echo "Compilation finished."
echo "----------------------------------------"

# --- Running MLP Forward Pass ---
echo "Running MLP Forward Pass on DCU..."
# Capture the full output for detailed parsing by the Python script
${OUTPUT_DIR}/${MLP_EXEC_NAME} > ${RESULTS_DIR}/mlp_raw_output.txt
# Copy the raw output to the file expected by the visualization script
# The Python script will parse the necessary lines from this file.
cp ${RESULTS_DIR}/mlp_raw_output.txt ${MLP_GPU_TIME_FILE}
# If you still want to see the output on console during the run, use tee and then copy:
# ${OUTPUT_DIR}/${MLP_EXEC_NAME} | tee ${RESULTS_DIR}/mlp_raw_output.txt
# cp ${RESULTS_DIR}/mlp_raw_output.txt ${MLP_GPU_TIME_FILE}

echo "MLP GPU detailed output saved to ${MLP_GPU_TIME_FILE}"
echo "----------------------------------------"

# --- Running CPU Matmul Tests ---
# Note: The C++ program sourcefile.cpp appends to 'cpu_performance_data.csv' in its execution directory.
# We will move it to RESULTS_DIR after all CPU tests.

echo "Running CPU Matmul Baseline..."
${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} baseline
echo "Running CPU Matmul OpenMP..."
${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} openmp
echo "Running CPU Matmul Block Tiling (bs=32)..."
${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} block 32
echo "Running CPU Matmul Block Tiling Improved (bs=32)..."
${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} block_improved 32
# Add more block sizes or other CPU methods if needed

echo "Running CPU Matmul MPI (4 processes)..."
mpirun --allow-run-as-root -np 4 ${OUTPUT_DIR}/${CPU_MATMUL_EXEC_NAME} mpi

# Move the generated CPU performance data
if [ -f "cpu_performance_data.csv" ]; then
    mv cpu_performance_data.csv ${CPU_PERF_CSV}
    echo "CPU performance data saved to ${CPU_PERF_CSV}"
else
    echo "Warning: cpu_performance_data.csv not found after CPU tests."
fi
echo "----------------------------------------"

# --- Running DCU Matmul Baseline Test ---
# Note: The C++ program sourcefile_dcu.cpp appends to 'dcu_performance_data.csv' in its execution directory.
# We will move it to RESULTS_DIR.
echo "Running DCU Matmul Baseline..."
${OUTPUT_DIR}/${DCU_MATMUL_EXEC_NAME}

# Move the generated DCU performance data
if [ -f "dcu_performance_data.csv" ]; then
    mv dcu_performance_data.csv ${DCU_PERF_CSV}
    echo "DCU performance data saved to ${DCU_PERF_CSV}"
else
    echo "Warning: dcu_performance_data.csv not found after DCU tests."
fi
echo "----------------------------------------"

# --- Theoretical Analysis & Tool Usage Reminder ---
echo "PERFORMANCE ANALYSIS NOTES:"
echo "1. Use 'rocm-smi' in a separate terminal during DCU program execution to monitor utilization."
echo "   Example: watch -n 1 rocm-smi"
echo "2. Use 'rocgdb ${OUTPUT_DIR}/${MLP_EXEC_NAME}' for debugging DCU kernels if needed (if applicable)."
echo "----------------------------------------"


# --- Running Python Visualization Script ---
echo "Running Python visualization script..."
python3 ${PYTHON_VIS_SCRIPT} ${MLP_GPU_TIME_FILE} ${CPU_PERF_CSV} ${DCU_PERF_CSV} ${RESULTS_DIR}
echo "Visualization script finished. Plots should be in ${RESULTS_DIR}"
echo "----------------------------------------"
echo "All tests completed."

# Make the script executable: chmod +x run_performance_tests.sh
