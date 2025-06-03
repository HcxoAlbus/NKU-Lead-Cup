import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re

def plot_cpu_performance(cpu_csv_file, output_dir):
    """Plots CPU matrix multiplication performance."""
    if not os.path.exists(cpu_csv_file):
        print(f"Warning: CPU performance file not found: {cpu_csv_file}")
        return
    try:
        df_cpu = pd.read_csv(cpu_csv_file)
        if df_cpu.empty:
            print(f"Warning: CPU performance file is empty: {cpu_csv_file}")
            return

        # Ensure 'Time' is numeric, convert if necessary
        df_cpu['Time'] = pd.to_numeric(df_cpu['Time'], errors='coerce')
        df_cpu.dropna(subset=['Time'], inplace=True)


        plt.figure(figsize=(12, 7))
        bars = plt.bar(df_cpu['Method'], df_cpu['Time'], color=['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral', 'cyan'])
        plt.xlabel("CPU Method")
        plt.ylabel("Time (seconds)")
        plt.title("CPU Matrix Multiplication Performance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * df_cpu['Time'].max(), f'{yval:.4f}', ha='center', va='bottom')

        plot_filename = os.path.join(output_dir, "cpu_performance_comparison.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"CPU performance plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error plotting CPU performance: {e}")

def plot_dcu_matmul_performance(dcu_csv_file, output_dir):
    """Plots DCU matrix multiplication performance against its CPU baseline."""
    if not os.path.exists(dcu_csv_file):
        print(f"Warning: DCU performance file not found: {dcu_csv_file}")
        return
    try:
        df_dcu = pd.read_csv(dcu_csv_file)
        if df_dcu.empty:
            print(f"Warning: DCU performance file is empty: {dcu_csv_file}")
            return
        
        df_dcu['Time'] = pd.to_numeric(df_dcu['Time'], errors='coerce')
        df_dcu.dropna(subset=['Time'], inplace=True)

        # Filter for relevant methods for a clear comparison
        # Example: Compare CPU_Baseline_for_DCU_comparison, HIP_Kernel, HIP_Total_Incl_Memcpy
        methods_to_plot = ['CPU_Baseline_for_DCU_comparison', 'HIP_Kernel', 'HIP_Total_Incl_Memcpy']
        df_dcu_filtered = df_dcu[df_dcu['Method'].isin(methods_to_plot)]
        
        if df_dcu_filtered.empty:
            print(f"Warning: No relevant methods found in DCU performance file for plotting: {dcu_csv_file}")
            return

        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_dcu_filtered['Method'], df_dcu_filtered['Time'], color=['blue', 'red', 'orange'])
        plt.xlabel("Method (DCU Matmul vs CPU)")
        plt.ylabel("Time (seconds)")
        plt.title("DCU Matrix Multiplication Performance (N=1024, M=2048, P=512)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * df_dcu_filtered['Time'].max(), f'{yval:.4f}', ha='center', va='bottom')

        plot_filename = os.path.join(output_dir, "dcu_matmul_performance_comparison.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"DCU matmul performance plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error plotting DCU matmul performance: {e}")

def parse_mlp_gpu_times(content, custom_kernel_header, hipblas_header):
    """Parses MLP GPU times for a specific section (custom or hipBLAS)."""
    data = {
        "total_time_s": None, "layer1_time_s": None, "layer2_time_s": None,
        "l1_matmul_s": None, "l1_bias_relu_s": None, 
        "l2_matmul_s": None, "l2_bias_s": None,
        "overhead_s": None, "valid": False
    }
    
    # Find the relevant block of text
    if custom_kernel_header in content:
        block_match = re.search(f"{re.escape(custom_kernel_header)}(.*?)(?=\n===|\Z)", content, re.DOTALL)
        if not block_match: return data
        block_content = block_match.group(1)

        total_match = re.search(r'GPU MLP \(Custom Kernels\) Forward Pass Time: ([\d.]+) ms', block_content)
        l1_match = re.search(r'Hidden Layer \(Custom Fused\) Time: ([\d.]+) ms', block_content)
        l2_match = re.search(r'Output Layer \(Custom Fused\) Time: ([\d.]+) ms', block_content)
        overhead_match = re.search(r'Memory Transfer Overhead \(Custom\): ([\d.]+) ms', block_content)

        if total_match and l1_match and l2_match:
            data["total_time_s"] = float(total_match.group(1)) / 1000.0
            data["layer1_time_s"] = float(l1_match.group(1)) / 1000.0
            data["layer2_time_s"] = float(l2_match.group(1)) / 1000.0
            if overhead_match:
                 data["overhead_s"] = float(overhead_match.group(1)) / 1000.0
            data["valid"] = True

    elif hipblas_header in content:
        block_match = re.search(f"{re.escape(hipblas_header)}(.*?)(?=\n===|\Z)", content, re.DOTALL)
        if not block_match: return data
        block_content = block_match.group(1)

        total_match = re.search(r'GPU MLP \(hipBLAS path\) Forward Pass Time: ([\d.]+) ms', block_content)
        l1_combined_match = re.search(r'Layer 1 Combined \(MatMul \+ Bias/ReLU\) Time: ([\d.]+) ms', block_content)
        l2_combined_match = re.search(r'Layer 2 Combined \(MatMul \+ Bias\) Time: ([\d.]+) ms', block_content)
        
        l1_matmul_match = re.search(r'Layer 1 \(hipBLAS MatMul\) Time: ([\d.]+) ms', block_content)
        l1_bias_relu_match = re.search(r'Layer 1 \(Custom Bias/ReLU\) Time: ([\d.]+) ms', block_content)
        l2_matmul_match = re.search(r'Layer 2 \(hipBLAS MatMul\) Time: ([\d.]+) ms', block_content)
        l2_bias_match = re.search(r'Layer 2 \(Custom Bias\) Time: ([\d.]+) ms', block_content)
        overhead_match = re.search(r'Memory Transfer Overhead \(hipBLAS path\): ([\d.]+) ms', block_content)

        if total_match and l1_combined_match and l2_combined_match:
            data["total_time_s"] = float(total_match.group(1)) / 1000.0
            data["layer1_time_s"] = float(l1_combined_match.group(1)) / 1000.0 # Combined L1
            data["layer2_time_s"] = float(l2_combined_match.group(1)) / 1000.0 # Combined L2
            if l1_matmul_match: data["l1_matmul_s"] = float(l1_matmul_match.group(1)) / 1000.0
            if l1_bias_relu_match: data["l1_bias_relu_s"] = float(l1_bias_relu_match.group(1)) / 1000.0
            if l2_matmul_match: data["l2_matmul_s"] = float(l2_matmul_match.group(1)) / 1000.0
            if l2_bias_match: data["l2_bias_s"] = float(l2_bias_match.group(1)) / 1000.0
            if overhead_match: data["overhead_s"] = float(overhead_match.group(1)) / 1000.0
            data["valid"] = True
    else: # Fallback for old single value format
        try:
            data["total_time_s"] = float(content.strip()) / 1000.0
            data["valid"] = True # Mark as valid but with limited info
        except ValueError:
            pass # Not a simple float
            
    return data


def plot_mlp_gpu_vs_cpu_baseline(mlp_gpu_time_file, cpu_csv_file, output_dir):
    """Plots MLP GPU time (both custom and hipBLAS paths) against a relevant CPU baseline with detailed breakdown."""
    if not os.path.exists(mlp_gpu_time_file):
        print(f"Warning: MLP GPU time file not found: {mlp_gpu_time_file}")
        return
    if not os.path.exists(cpu_csv_file):
        print(f"Warning: CPU performance file not found for MLP comparison: {cpu_csv_file}")
        return

    try:
        with open(mlp_gpu_time_file, 'r') as f:
            content = f.read()

        custom_header = "=== Performance Analysis (Custom Fused Kernels) ==="
        hipblas_header = "=== Performance Analysis (hipBLAS + Custom Bias/Activation Kernels) ==="
            
        custom_data = parse_mlp_gpu_times(content, custom_header, None)
        hipblas_data = parse_mlp_gpu_times(content, None, hipblas_header)

        if not custom_data["valid"] and not hipblas_data["valid"]:
             # Try parsing as old format if specific headers not found
            old_format_data = parse_mlp_gpu_times(content, None, None)
            if old_format_data["valid"]:
                custom_data = old_format_data # Assign to custom for plotting
                print("Warning: Parsed MLP GPU time in old single-value format. Detailed breakdown might be limited.")
            else:
                print(f"Error: Could not parse MLP GPU times from {mlp_gpu_time_file}. Ensure the format is correct.")
                return
        
        df_cpu = pd.read_csv(cpu_csv_file)
        cpu_baseline_time_series = df_cpu[df_cpu['Method'] == 'Baseline']['Time']
        if cpu_baseline_time_series.empty:
            print("Warning: 'Baseline' method not found in CPU performance data for MLP comparison.")
            cpu_baseline_time_s = None
        else:
            cpu_baseline_time_s = pd.to_numeric(cpu_baseline_time_series.iloc[0], errors='coerce')
            if pd.isna(cpu_baseline_time_s):
                 print("Warning: 'Baseline' CPU time is not a valid number.")
                 cpu_baseline_time_s = None

        fig_height = 7
        num_subplots = 1
        if custom_data["valid"] and custom_data["layer1_time_s"] is not None:
            num_subplots +=1
        if hipblas_data["valid"] and hipblas_data["layer1_time_s"] is not None:
            if not (custom_data["valid"] and custom_data["layer1_time_s"] is not None) : # only add if not already added
                 num_subplots +=1
            elif custom_data["valid"] and custom_data["layer1_time_s"] is not None and hipblas_data["valid"] and hipblas_data["layer1_time_s"] is not None:
                 num_subplots = 3 # main, custom_breakdown, hipblas_breakdown


        fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, fig_height), squeeze=False)
        ax_idx = 0
        
        # Subplot 1: Overall Comparison
        ax1 = axes[0, ax_idx]
        ax_idx +=1
        methods = []
        times = []
        colors = []

        if cpu_baseline_time_s is not None:
            methods.append('CPU Matmul Baseline (Ref)')
            times.append(cpu_baseline_time_s)
            colors.append('gray')
        
        if custom_data["valid"] and custom_data["total_time_s"] is not None:
            methods.append('MLP GPU (Custom Kernels)')
            times.append(custom_data["total_time_s"])
            colors.append('green')

        if hipblas_data["valid"] and hipblas_data["total_time_s"] is not None:
            methods.append('MLP GPU (hipBLAS Path)')
            times.append(hipblas_data["total_time_s"])
            colors.append('purple')
        
        if not methods: # If no data to plot
            ax1.text(0.5, 0.5, 'No MLP GPU data found', ha='center', va='center', transform=ax1.transAxes)
        else:
            bars1 = ax1.bar(methods, times, color=colors)
            ax1.set_ylabel("Time (seconds)")
            ax1.set_title("MLP GPU vs CPU Baseline")
            ax1.tick_params(axis='x', rotation=20, ha="right")
            for bar in bars1:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * (max(times) if times else 0.001), 
                        f'{yval:.6f}', ha='center', va='bottom', fontsize=9)

        # Subplot 2: Custom Kernels GPU Layer Breakdown
        if custom_data["valid"] and custom_data["layer1_time_s"] is not None and custom_data["layer2_time_s"] is not None:
            ax2 = axes[0, ax_idx]
            ax_idx +=1
            gpu_methods = ['Hidden Layer\n(Fused)', 'Output Layer\n(Fused)', 'Mem Overhead', 'Total GPU']
            gpu_times = [custom_data["layer1_time_s"], custom_data["layer2_time_s"], 
                         custom_data.get("overhead_s") or 0.0, custom_data["total_time_s"]]
            gpu_colors = ['lightblue', 'lightcoral', 'lightsalmon', 'green']
            
            bars2 = ax2.bar(gpu_methods, gpu_times, color=gpu_colors)
            ax2.set_ylabel("Time (seconds)")
            ax2.set_title("GPU MLP Breakdown (Custom Kernels)")
            ax2.tick_params(axis='x', rotation=0)
            for bar in bars2:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * (max(gpu_times) if gpu_times else 0.001), 
                        f'{yval:.6f}', ha='center', va='bottom', fontsize=9)
        elif num_subplots > 1 and ax_idx == 1 : # if we expected this plot but data was missing
            ax_missing = axes[0, ax_idx]
            ax_idx +=1
            ax_missing.text(0.5, 0.5, 'Custom Kernel Breakdown\nnot available', ha='center', va='center', 
                           transform=ax_missing.transAxes, fontsize=10)
            ax_missing.set_title("GPU Breakdown (Custom) N/A")


        # Subplot 3: hipBLAS Path GPU Layer Breakdown
        if hipblas_data["valid"] and hipblas_data["layer1_time_s"] is not None and hipblas_data["layer2_time_s"] is not None:
            ax3 = axes[0, ax_idx]
            ax_idx +=1
            hip_gpu_methods = [
                'L1 MatMul\n(hipBLAS)', 'L1 Bias+ReLU\n(Custom)', 
                'L2 MatMul\n(hipBLAS)', 'L2 Bias\n(Custom)', 'Mem Overhead', 'Total GPU']
            hip_gpu_times = [
                hipblas_data.get("l1_matmul_s") or 0.0, hipblas_data.get("l1_bias_relu_s") or 0.0,
                hipblas_data.get("l2_matmul_s") or 0.0, hipblas_data.get("l2_bias_s") or 0.0,
                hipblas_data.get("overhead_s") or 0.0, hipblas_data["total_time_s"]]
            hip_gpu_colors = ['deepskyblue', 'skyblue', 'salmon', 'lightcoral', 'lightsalmon', 'purple']
            
            bars3 = ax3.bar(hip_gpu_methods, hip_gpu_times, color=hip_gpu_colors)
            ax3.set_ylabel("Time (seconds)")
            ax3.set_title("GPU MLP Breakdown (hipBLAS Path)")
            ax3.tick_params(axis='x', rotation=20, ha="right")
            for bar in bars3:
                yval = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * (max(hip_gpu_times) if hip_gpu_times else 0.001), 
                        f'{yval:.6f}', ha='center', va='bottom', fontsize=9)
        elif num_subplots > 1 and ax_idx > 0 and ax_idx < num_subplots: # if we expected this plot but data was missing
            ax_missing = axes[0, ax_idx]
            ax_idx +=1
            ax_missing.text(0.5, 0.5, 'hipBLAS Path Breakdown\nnot available', ha='center', va='center', 
                           transform=ax_missing.transAxes, fontsize=10)
            ax_missing.set_title("GPU Breakdown (hipBLAS) N/A")


        plt.tight_layout(pad=2.0)
        plot_filename = os.path.join(output_dir, "mlp_gpu_vs_cpu_baseline_comparison.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"MLP GPU comparison plot saved to {plot_filename}")

    except Exception as e:
        print(f"Error plotting MLP GPU vs CPU baseline: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python vis_performance.py <mlp_gpu_time_file> <cpu_perf_csv> <dcu_perf_csv> <output_plot_dir>")
        sys.exit(1)

    mlp_gpu_time_file_arg = sys.argv[1]
    cpu_csv_file_arg = sys.argv[2]
    dcu_csv_file_arg = sys.argv[3]
    output_plot_dir_arg = sys.argv[4]

    if not os.path.exists(output_plot_dir_arg):
        os.makedirs(output_plot_dir_arg, exist_ok=True)

    print(f"--- Generating Performance Plots ---")
    plot_cpu_performance(cpu_csv_file_arg, output_plot_dir_arg)
    plot_dcu_matmul_performance(dcu_csv_file_arg, output_plot_dir_arg)
    plot_mlp_gpu_vs_cpu_baseline(mlp_gpu_time_file_arg, cpu_csv_file_arg, output_plot_dir_arg)
    print(f"--- Plot generation complete. Check the '{output_plot_dir_arg}' directory. ---")

