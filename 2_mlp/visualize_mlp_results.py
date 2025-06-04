import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from datetime import datetime

def parse_log_file(log_file_path):
    """Parses the MLP execution log file to extract performance data."""
    epochs_data = []
    training_summary = {}
    inference_summary = {}
    predictions_data = []

    # 修复正则表达式以匹配实际的日志格式
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
    """Plots average loss per epoch."""
    if epochs_df.empty:
        print("[WARN] No epoch data to plot for loss vs epoch.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_df["epoch"], epochs_df["loss"], marker='o', linestyle='-')
    plt.title("Average Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "loss_vs_epoch.png"))
    print(f"[PLOT] Saved loss vs epoch plot to {os.path.join(output_dir, 'loss_vs_epoch.png')}")
    plt.close()

def plot_epoch_time(epochs_df, output_dir="test_results/plots"):
    """Plots time taken per epoch."""
    if epochs_df.empty:
        print("[WARN] No epoch data to plot for epoch time.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_df["epoch"], epochs_df["time_ms"], marker='s', linestyle='--', color='r')
    plt.title("Time per Training Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (ms)")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "epoch_time.png"))
    print(f"[PLOT] Saved epoch time plot to {os.path.join(output_dir, 'epoch_time.png')}")
    plt.close()

def plot_performance_summary(training_summary, inference_summary, output_dir="test_results/plots"):
    """Plots summary of training and inference performance."""
    labels = ["Training", "Inference"]
    times_ms = [training_summary.get("total_time_ms", 0), inference_summary.get("total_time_ms", 0)]
    throughputs_sps = [training_summary.get("throughput_sps", 0), inference_summary.get("throughput_sps", 0)]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Total Time (ms)', color=color)
    bars = ax1.bar(labels, times_ms, color=color, alpha=0.6, width=0.4, label='Total Time (ms)')
    ax1.tick_params(axis='y', labelcolor=color)
    for bar in bars: # Add text labels on bars
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(times_ms), f'{yval:.0f} ms', ha='center', va='bottom')


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Throughput (samples/sec)', color=color)  # we already handled the x-label with ax1
    ax2.plot(labels, throughputs_sps, color=color, marker='o', linestyle=':', linewidth=2, markersize=8, label='Throughput (samples/sec)')
    ax2.tick_params(axis='y', labelcolor=color)
    for i, txt in enumerate(throughputs_sps):
         ax2.annotate(f'{txt:.2f} sps', (labels[i], throughputs_sps[i]), textcoords="offset points", xytext=(0,10), ha='center', color=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Training and Inference Performance Summary")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "performance_summary.png"))
    print(f"[PLOT] Saved performance summary plot to {os.path.join(output_dir, 'performance_summary.png')}")
    plt.close()

def plot_mse_summary(inference_summary, output_dir="test_results/plots"):
    """Plots MSE on test set."""
    mse_norm = inference_summary.get("mse_normalized", float('nan'))
    mse_denorm = inference_summary.get("mse_denormalized", float('nan'))

    if pd.isna(mse_norm) and pd.isna(mse_denorm):
        print("[WARN] No MSE data to plot.")
        return

    labels = ['MSE (Normalized)', 'MSE (Denormalized)']
    values = [mse_norm, mse_denorm]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['lightcoral', 'lightskyblue'])
    plt.title('Mean Squared Error (MSE) on Test Set')
    plt.ylabel('MSE Value')
    plt.yscale('log') # MSE values can vary a lot, log scale might be better
    for bar in bars:
        yval = bar.get_height()
        if not pd.isna(yval):
             plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4e}', ha='center', va='bottom')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "mse_summary.png"))
    print(f"[PLOT] Saved MSE summary plot to {os.path.join(output_dir, 'mse_summary.png')}")
    plt.close()

def plot_predictions(predictions_df, output_dir="test_results/plots"):
    """Plots a sample of predicted vs actual values."""
    if predictions_df.empty:
        print("[WARN] No prediction data to plot.")
        return
    
    num_samples_to_plot = min(len(predictions_df), 50) # Plot up to 50 samples
    plot_df = predictions_df.head(num_samples_to_plot)

    plt.figure(figsize=(12, 7))
    plt.plot(plot_df.index, plot_df['actual'], label='Actual Values', marker='x', linestyle='-')
    plt.plot(plot_df.index, plot_df['predicted'], label='Predicted Values', marker='.', linestyle='--')
    plt.title(f'Sample of Denormalized Predictions vs Actual (First {num_samples_to_plot} Test Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Bandwidth Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "predictions_vs_actual.png"))
    print(f"[PLOT] Saved predictions vs actual plot to {os.path.join(output_dir, 'predictions_vs_actual.png')}")
    plt.close()

def plot_train_val_loss(epochs_df, output_dir="test_results/plots"):
    """Plots training and validation loss comparison."""
    if epochs_df.empty:
        print("[WARN] No epoch data to plot for train/val loss.")
        return
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_df["epoch"], epochs_df["train_loss"], marker='o', linestyle='-', label='Training Loss', alpha=0.7)
    plt.plot(epochs_df["epoch"], epochs_df["val_loss"], marker='s', linestyle='--', label='Validation Loss', alpha=0.7)
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "train_val_loss.png"))
    print(f"[PLOT] Saved train/val loss plot to {os.path.join(output_dir, 'train_val_loss.png')}")
    plt.close()

def plot_learning_rate_schedule(epochs_df, output_dir="test_results/plots"):
    """Plots learning rate schedule over epochs."""
    if epochs_df.empty or 'learning_rate' not in epochs_df.columns:
        print("[WARN] No learning rate data to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_df["epoch"], epochs_df["learning_rate"], marker='o', linestyle='-', color='green')
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "learning_rate_schedule.png"))
    print(f"[PLOT] Saved learning rate schedule plot to {os.path.join(output_dir, 'learning_rate_schedule.png')}")
    plt.close()

def plot_overfitting_analysis(epochs_df, output_dir="test_results/plots"):
    """Plots overfitting analysis showing train/val loss gap."""
    if epochs_df.empty or 'train_loss' not in epochs_df.columns or 'val_loss' not in epochs_df.columns:
        print("[WARN] No train/val loss data for overfitting analysis.")
        return
    
    # Calculate loss difference (val_loss - train_loss)
    loss_gap = epochs_df["val_loss"] - epochs_df["train_loss"]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_df["epoch"], epochs_df["train_loss"], label='Training Loss', color='blue')
    plt.plot(epochs_df["epoch"], epochs_df["val_loss"], label='Validation Loss', color='red')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_df["epoch"], loss_gap, color='orange', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title("Overfitting Gap (Val Loss - Train Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Difference")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "overfitting_analysis.png"))
    print(f"[PLOT] Saved overfitting analysis plot to {os.path.join(output_dir, 'overfitting_analysis.png')}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MLP performance results from a log file.")
    parser.add_argument("log_file", help="Path to the log file generated by run_mlp_test.sh")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"[ERROR] Log file not found: {args.log_file}")
        exit(1)

    print(f"[INFO] Parsing log file: {args.log_file}")
    epochs_data, training_summary, inference_summary, predictions_data = parse_log_file(args.log_file)

    # Create a directory for plots based on the log file name
    log_file_basename = os.path.splitext(os.path.basename(args.log_file))[0]
    plot_output_dir = os.path.join(os.path.dirname(args.log_file) or ".", "plots", log_file_basename) # Place plots in a subfolder related to the log
    
    print(f"[INFO] Plots will be saved in: {plot_output_dir}")

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
    
    print("[INFO] Visualization script finished.")

