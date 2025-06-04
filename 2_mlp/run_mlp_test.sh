#!/bin/bash
# filepath: d:\sec_semester_code\CPP\Lead_cup\2_mlp\run_mlp_test.sh

# 定义文件名和路径
SOURCE_FILE="sourcefile_mlp.cpp"
EXECUTABLE_NAME="mlp_test_run"
LOG_DIR="test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_output_${TIMESTAMP}.log"
CSV_REPORT_FILE="${LOG_DIR}/performance_summary_${TIMESTAMP}.csv"
CPP_FILE_PATH="${SOURCE_FILE}" # 与C++内路径匹配，用于编译

# 创建日志目录
mkdir -p ${LOG_DIR}

echo "-----------------------------------------------------"
echo "MLP Performance Test Script"
echo "-----------------------------------------------------"
echo

# 步骤 1: 编译 C++ 代码
echo "[INFO] Compiling ${SOURCE_FILE}..."
hipcc "${CPP_FILE_PATH}" -o "${EXECUTABLE_NAME}"
if [ $? -ne 0 ]; then
    echo "[ERROR] Compilation failed. Exiting."
    exit 1
fi
echo "[INFO] Compilation successful: ${EXECUTABLE_NAME}"
echo

# 步骤 2: 执行 MLP 网络并记录输出
echo "[INFO] Running MLP network..."
echo "[INFO] Output will be saved to: ${LOG_FILE}"
# 使用 tee 同时输出到控制台和文件
./${EXECUTABLE_NAME} 2>&1 | tee "${LOG_FILE}"
# 等待后台的tee完成写入
wait
echo
echo "[INFO] MLP execution finished."
echo

# 步骤 3: 从日志文件中提取性能指标并保存到 CSV
echo "[INFO] Extracting performance metrics to ${CSV_REPORT_FILE}..."

# 初始化 CSV 文件并写入表头
echo "Timestamp,TotalTrainingTime_ms,TrainingThroughput_sps,TotalInferenceTime_ms,InferenceThroughput_sps,MSE_Normalized_Test,MSE_Denormalized_Test,TotalExecutionTime_ms,LogFile" > "${CSV_REPORT_FILE}"

# 从日志文件中提取数据 - 修复awk字段索引
TRAIN_TIME=$(grep "Training finished. Total time:" "${LOG_FILE}" | awk '{print $5}')
TRAIN_THROUGHPUT=$(grep "Training throughput:" "${LOG_FILE}" | awk '{print $3}')
INFER_TIME=$(grep "Inference finished. Total time:" "${LOG_FILE}" | awk '{print $5}')
INFER_THROUGHPUT=$(grep "Inference throughput:" "${LOG_FILE}" | awk '{print $3}')
MSE_NORM=$(grep "Mean Squared Error (MSE) on (normalized) Test Set:" "${LOG_FILE}" | awk '{print $10}')
MSE_DENORM=$(grep "Mean Squared Error (MSE) on (denormalized) Test Set:" "${LOG_FILE}" | awk '{print $10}')
TOTAL_EXEC_TIME=$(grep "Total execution time:" "${LOG_FILE}" | awk '{print $4}')

# 检查是否所有值都已提取
if [ -z "$TRAIN_TIME" ] || [ -z "$TRAIN_THROUGHPUT" ] || [ -z "$INFER_TIME" ] || [ -z "$INFER_THROUGHPUT" ] || [ -z "$MSE_NORM" ] || [ -z "$MSE_DENORM" ] || [ -z "$TOTAL_EXEC_TIME" ]; then
    echo "[WARNING] Could not extract all performance metrics. CSV might be incomplete."
    TRAIN_TIME=${TRAIN_TIME:-"N/A"}
    TRAIN_THROUGHPUT=${TRAIN_THROUGHPUT:-"N/A"}
    INFER_TIME=${INFER_TIME:-"N/A"}
    INFER_THROUGHPUT=${INFER_THROUGHPUT:-"N/A"}
    MSE_NORM=${MSE_NORM:-"N/A"}
    MSE_DENORM=${MSE_DENORM:-"N/A"}
    TOTAL_EXEC_TIME=${TOTAL_EXEC_TIME:-"N/A"}
fi

# 将数据写入 CSV 文件
echo "${TIMESTAMP},${TRAIN_TIME},${TRAIN_THROUGHPUT},${INFER_TIME},${INFER_THROUGHPUT},${MSE_NORM},${MSE_DENORM},${TOTAL_EXEC_TIME},run_output_${TIMESTAMP}.log" >> "${CSV_REPORT_FILE}"

echo "[INFO] Performance metrics saved to ${CSV_REPORT_FILE}"
echo

# 步骤 4: 自动运行 Python 可视化脚本
echo "[INFO] Running Python script to visualize results from ${LOG_FILE}..."
# 确保 Python 环境和必要的库 (matplotlib, pandas) 已安装
# 假设 visualize_mlp_results.py 与此脚本在同一目录
PYTHON_SCRIPT_PATH="./visualize_mlp_results.py" # 或者指定绝对路径

if [ -f "${PYTHON_SCRIPT_PATH}" ]; then
    python "${PYTHON_SCRIPT_PATH}" "${LOG_FILE}"
    if [ $? -ne 0 ]; then
        echo "[WARNING] Python visualization script encountered an error."
    else
        echo "[INFO] Python visualization script finished successfully."
    fi
else
    echo "[WARNING] Python visualization script not found at ${PYTHON_SCRIPT_PATH}. Skipping visualization."
fi

echo
echo "-----------------------------------------------------"
echo "Script finished."
echo "-----------------------------------------------------"

# 清理可执行文件 (可选)
# rm -f ${EXECUTABLE_NAME}