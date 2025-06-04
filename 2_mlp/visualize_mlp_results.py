import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from datetime import datetime

def parse_log_file(log_file_path):
    """解析MLP执行日志文件以提取性能数据。"""
    epochs_data = []
    training_summary = {}
    inference_summary = {}
    predictions_data = []


    epoch_pattern = re.compile(r"\[Epoch\s+(\d+)/\d+\] Train Loss: ([\d.eE+-]+), Val Loss: ([\d.eE+-]+), LR: ([\d.eE+-]+), Time: (\d+) ms")
    early_stop_pattern = re.compile(r"\[INFO\] Early stopping triggered at epoch (\d+). Best validation loss: ([\d.eE+-]+)")
    train_time_pattern = re.compile(r"\[INFO\] Training finished. Total time: (\d+) ms.")
    train_throughput_pattern = re.compile(r"\[INFO\] Training throughput: ([\d.]+) samples/sec.")
    infer_time_pattern = re.compile(r"\[INFO\] Inference finished. Total time: ([\d.]+) ms.") # 允许浮点数
    infer_throughput_pattern = re.compile(r"\[INFO\] Inference throughput: ([\d.]+) samples/sec.")
    mse_norm_pattern = re.compile(r"\[INFO\] Mean Squared Error \(MSE\) on \(normalized\) Test Set: ([\d.eE+-]+)")
    mse_denorm_pattern = re.compile(r"\[INFO\] Mean Squared Error \(MSE\) on \(denormalized\) Test Set: ([\d.eE+-]+)")
    prediction_pattern = re.compile(r"Sample\s+(\d+):\s+Predicted:\s+([\d.]+),\s+Actual:\s+([\d.]+)")
    total_exec_time_pattern = re.compile(r"\[INFO\] Total execution time: (\d+) ms.")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = epoch_pattern.search(line)
            if match:
                epochs_data.append({
                    "epoch": int(match.group(1)),
                    "train_loss": float(match.group(2)),
                    "val_loss": float(match.group(3)),
                    "learning_rate": float(match.group(4)),
                    "time_ms": int(match.group(5))
                })
                continue

            match = early_stop_pattern.search(line)
            if match:
                training_summary["early_stop_epoch"] = int(match.group(1))
                training_summary["best_val_loss"] = float(match.group(2))
                continue

            match = train_time_pattern.search(line)
            if match:
                training_summary["total_time_ms"] = int(match.group(1))
                continue
            
            match = train_throughput_pattern.search(line)
            if match:
                training_summary["throughput_sps"] = float(match.group(1))
                continue

            match = infer_time_pattern.search(line)
            if match:
                inference_summary["total_time_ms"] = float(match.group(1)) # 解析为浮点数
                continue

            match = infer_throughput_pattern.search(line)
            if match:
                inference_summary["throughput_sps"] = float(match.group(1))
                continue
            
            match = mse_norm_pattern.search(line)
            if match:
                inference_summary["mse_normalized"] = float(match.group(1))
                continue

            match = mse_denorm_pattern.search(line)
            if match:
                inference_summary["mse_denormalized"] = float(match.group(1))
                continue
            
            match = prediction_pattern.search(line)
            if match:
                predictions_data.append({
                    "predicted": float(match.group(2)),
                    "actual": float(match.group(3))
                })
                continue
            
            match = total_exec_time_pattern.search(line)
            if match:
                training_summary["overall_exec_time_ms"] = int(match.group(1))
                inference_summary["overall_exec_time_ms"] = int(match.group(1))
                continue

    return epochs_data, training_summary, inference_summary, predictions_data

def plot_loss_vs_epoch(epochs_df, output_dir="test_results/plots"):
    """绘制每个epoch的平均损失。"""
    if epochs_df.empty:
        print("[WARN] 没有epoch数据可供绘制损失图表。")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_df["epoch"], epochs_df["loss"], marker='o', linestyle='-')
    plt.title("每个Epoch的平均训练损失")
    plt.xlabel("Epoch")
    plt.ylabel("平均损失")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "loss_vs_epoch.png"))
    print(f"[PLOT] 已保存损失-epoch图表到 {os.path.join(output_dir, 'loss_vs_epoch.png')}")
    plt.close()

def plot_epoch_time(epochs_df, output_dir="test_results/plots"):
    """绘制每个epoch所需的时间。"""
    if epochs_df.empty:
        print("[WARN] 没有epoch数据可供绘制时间图表。")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_df["epoch"], epochs_df["time_ms"], marker='s', linestyle='--', color='r')
    plt.title("每个训练Epoch的时间")
    plt.xlabel("Epoch")
    plt.ylabel("时间 (ms)")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "epoch_time.png"))
    print(f"[PLOT] 已保存epoch时间图表到 {os.path.join(output_dir, 'epoch_time.png')}")
    plt.close()

def plot_performance_summary(training_summary, inference_summary, output_dir="test_results/plots"):
    """绘制训练和推理性能摘要。"""
    labels = ["训练", "推理"]
    times_ms = [training_summary.get("total_time_ms", 0), inference_summary.get("total_time_ms", 0)]
    throughputs_sps = [training_summary.get("throughput_sps", 0), inference_summary.get("throughput_sps", 0)]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('阶段')
    ax1.set_ylabel('总时间 (ms)', color=color)
    bars = ax1.bar(labels, times_ms, color=color, alpha=0.6, width=0.4, label='总时间 (ms)')
    ax1.tick_params(axis='y', labelcolor=color)
    for bar in bars: # 在柱形图上添加文本标签
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(times_ms), f'{yval:.0f} ms', ha='center', va='bottom')


    ax2 = ax1.twinx()  # 实例化共享相同x轴的第二个坐标轴
    color = 'tab:red'
    ax2.set_ylabel('吞吐量 (样本/秒)', color=color)  # 我们已经用ax1处理了x标签
    ax2.plot(labels, throughputs_sps, color=color, marker='o', linestyle=':', linewidth=2, markersize=8, label='吞吐量 (样本/秒)')
    ax2.tick_params(axis='y', labelcolor=color)
    for i, txt in enumerate(throughputs_sps):
         ax2.annotate(f'{txt:.2f} sps', (labels[i], throughputs_sps[i]), textcoords="offset points", xytext=(0,10), ha='center', color=color)


    fig.tight_layout()  # 否则右侧y轴标签会被稍微裁剪
    plt.title("训练和推理性能总结")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "performance_summary.png"))
    print(f"[PLOT] 已保存性能总结图表到 {os.path.join(output_dir, 'performance_summary.png')}")
    plt.close()

def plot_mse_summary(inference_summary, output_dir="test_results/plots"):
    """绘制测试集上的MSE。"""
    mse_norm = inference_summary.get("mse_normalized", float('nan'))
    mse_denorm = inference_summary.get("mse_denormalized", float('nan'))

    if pd.isna(mse_norm) and pd.isna(mse_denorm):
        print("[WARN] 没有MSE数据可供绘图。")
        return

    labels = ['MSE (归一化)', 'MSE (反归一化)']
    values = [mse_norm, mse_denorm]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['lightcoral', 'lightskyblue'])
    plt.title('测试集上的均方误差(MSE)')
    plt.ylabel('MSE值')
    plt.yscale('log') # MSE值可能变化很大，对数尺度可能更好
    for bar in bars:
        yval = bar.get_height()
        if not pd.isna(yval):
             plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4e}', ha='center', va='bottom')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "mse_summary.png"))
    print(f"[PLOT] 已保存MSE总结图表到 {os.path.join(output_dir, 'mse_summary.png')}")
    plt.close()

def plot_predictions(predictions_df, output_dir="test_results/plots"):
    """绘制预测值与实际值的样本图。"""
    if predictions_df.empty:
        print("[WARN] 没有预测数据可供绘图。")
        return
    
    num_samples_to_plot = min(len(predictions_df), 50) # 最多绘制50个样本
    plot_df = predictions_df.head(num_samples_to_plot)

    plt.figure(figsize=(12, 7))
    plt.plot(plot_df.index, plot_df['actual'], label='实际值', marker='x', linestyle='-')
    plt.plot(plot_df.index, plot_df['predicted'], label='预测值', marker='.', linestyle='--')
    plt.title(f'反归一化预测值与实际值对比样本(前{num_samples_to_plot}个测试样本)')
    plt.xlabel('样本索引')
    plt.ylabel('带宽值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "predictions_vs_actual.png"))
    print(f"[PLOT] 已保存预测值与实际值对比图表到 {os.path.join(output_dir, 'predictions_vs_actual.png')}")
    plt.close()

def plot_train_val_loss(epochs_df, output_dir="test_results/plots"):
    """绘制训练和验证损失对比图。"""
    if epochs_df.empty:
        print("[WARN] 没有epoch数据可供绘制训练/验证损失图。")
        return
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_df["epoch"], epochs_df["train_loss"], marker='o', linestyle='-', label='训练损失', alpha=0.7)
    plt.plot(epochs_df["epoch"], epochs_df["val_loss"], marker='s', linestyle='--', label='验证损失', alpha=0.7)
    plt.title("训练损失与验证损失对比")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "train_val_loss.png"))
    print(f"[PLOT] 已保存训练/验证损失对比图表到 {os.path.join(output_dir, 'train_val_loss.png')}")
    plt.close()

def plot_learning_rate_schedule(epochs_df, output_dir="test_results/plots"):
    """绘制不同epoch下的学习率变化图。"""
    if epochs_df.empty or 'learning_rate' not in epochs_df.columns:
        print("[WARN] 没有学习率数据可供绘图。")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_df["epoch"], epochs_df["learning_rate"], marker='o', linestyle='-', color='green')
    plt.title("学习率变化图")
    plt.xlabel("Epoch")
    plt.ylabel("学习率")
    plt.yscale('log')  # 对数尺度以便更好地可视化
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "learning_rate_schedule.png"))
    print(f"[PLOT] 已保存学习率变化图表到 {os.path.join(output_dir, 'learning_rate_schedule.png')}")
    plt.close()

def plot_overfitting_analysis(epochs_df, output_dir="test_results/plots"):
    """绘制过拟合分析图，展示训练/验证损失间隙。"""
    if epochs_df.empty or 'train_loss' not in epochs_df.columns or 'val_loss' not in epochs_df.columns:
        print("[WARN] 没有训练/验证损失数据可供过拟合分析。")
        return
    
    # 计算损失差异 (val_loss - train_loss)
    loss_gap = epochs_df["val_loss"] - epochs_df["train_loss"]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_df["epoch"], epochs_df["train_loss"], label='训练损失', color='blue')
    plt.plot(epochs_df["epoch"], epochs_df["val_loss"], label='验证损失', color='red')
    plt.title("训练损失与验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_df["epoch"], loss_gap, color='orange', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title("过拟合间隙 (验证损失 - 训练损失)")
    plt.xlabel("Epoch")
    plt.ylabel("损失差异")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "overfitting_analysis.png"))
    print(f"[PLOT] 已保存过拟合分析图表到 {os.path.join(output_dir, 'overfitting_analysis.png')}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从日志文件中可视化MLP性能结果。")
    parser.add_argument("log_file", help="由run_mlp_test.sh生成的日志文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"[ERROR] 找不到日志文件: {args.log_file}")
        exit(1)

    print(f"[INFO] 正在解析日志文件: {args.log_file}")
    epochs_data, training_summary, inference_summary, predictions_data = parse_log_file(args.log_file)

    # 根据日志文件名创建图表目录
    log_file_basename = os.path.splitext(os.path.basename(args.log_file))[0]
    plot_output_dir = os.path.join(os.path.dirname(args.log_file) or ".", "plots", log_file_basename) # 将图表放在与日志相关的子文件夹中
    
    print(f"[INFO] 图表将保存在: {plot_output_dir}")

    epochs_df = pd.DataFrame(epochs_data)
    predictions_df = pd.DataFrame(predictions_data)

    # 原有的图表
    plot_loss_vs_epoch(epochs_df.rename(columns={'train_loss': 'loss'}), output_dir=plot_output_dir)  # 兼容性重命名
    plot_epoch_time(epochs_df, output_dir=plot_output_dir)
    plot_performance_summary(training_summary, inference_summary, output_dir=plot_output_dir)
    plot_mse_summary(inference_summary, output_dir=plot_output_dir)
    plot_predictions(predictions_df, output_dir=plot_output_dir)
    
    # 新增的图表
    plot_train_val_loss(epochs_df, output_dir=plot_output_dir)
    plot_learning_rate_schedule(epochs_df, output_dir=plot_output_dir)
    plot_overfitting_analysis(epochs_df, output_dir=plot_output_dir)
    
    print("[INFO] 可视化脚本已完成。")
