import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# 设置 matplotlib 使用默认字体，避免中文字体问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']  # 使用系统默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def plot_performance_comparison(methods, times, title="Performance Comparison", save_prefix="cpu_performance"):
    """
    Generate and save performance comparison bar chart for different optimization methods.

    Parameters:
    methods (list of str): List of optimization method names.
    times (list of float): List of execution times (in seconds).
    title (str): Chart title.
    save_prefix (str): File prefix for saving.
    """
    if not methods or not times or len(methods) != len(times):
        print("Error: Methods and times lists cannot be empty and must have the same length.")
        return

    x_pos = np.arange(len(methods))

    plt.figure(figsize=(15, 8)) # Increase image size to accommodate more data
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))  # Use colormap to generate enough colors
    bars = plt.bar(x_pos, times, align='center', alpha=0.7, color=colors)
    
    plt.xticks(x_pos, methods, rotation=45, ha="right")
    plt.ylabel('Execution Time (seconds)')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display values on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(times), f'{yval:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout() # Adjust layout to prevent label overlap
    
    # Save image
    filename = f"{save_prefix}_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Performance comparison chart saved as: {filename}")
    plt.close()  # Close figure to free memory

def plot_speedup_comparison(baseline_time, methods, times, title="Speedup Comparison (vs Baseline)", save_prefix="cpu_speedup"):
    """
    Generate and save speedup comparison bar chart relative to baseline.

    Parameters:
    baseline_time (float): Baseline method execution time.
    methods (list of str): List of optimization method names (excluding baseline).
    times (list of float): List of corresponding optimization method execution times.
    title (str): Chart title.
    save_prefix (str): File prefix for saving.
    """
    if baseline_time <= 0:
        print("Error: Baseline time must be positive.")
        return
    if not methods or not times or len(methods) != len(times):
        print("Error: Methods and times lists cannot be empty and must have the same length.")
        return

    speedups = [baseline_time / t if t > 0 else 0 for t in times] 
    x_pos = np.arange(len(methods))

    plt.figure(figsize=(15, 8)) # Increase image size
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))  # Use different colormap
    bars = plt.bar(x_pos, speedups, align='center', alpha=0.7, color=colors)
    
    plt.xticks(x_pos, methods, rotation=45, ha="right")
    plt.ylabel('Speedup (Baseline Time / Method Time)')
    plt.title(title)
    plt.axhline(1, color='red', linestyle='--', linewidth=1.0, alpha=0.8, label='No speedup reference line (1x)') # Reference line for speedup = 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(speedups), f'{yval:.2f}x', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save image
    filename = f"{save_prefix}_speedup.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Speedup comparison chart saved as: {filename}")
    plt.close()  # Close figure to free memory

def read_performance_data(csv_filepath):
    """
    Read performance data from specified CSV file.
    Returns a dictionary with method names as keys and latest execution times as values.
    """
    data = {}
    if not os.path.exists(csv_filepath):
        print(f"Warning: CSV file not found: {csv_filepath}")
        return data
        
    try:
        with open(csv_filepath, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader, None) # Skip header
            if header is None or [h.strip() for h in header] != ['Method', 'Time']:
                print(f"Warning: {csv_filepath} header format incorrect. Expected 'Method,Time'")
                # Try to continue reading, assuming two-column format
            
            for row in reader:
                if len(row) == 2:
                    method = row[0].strip()
                    try:
                        time = float(row[1].strip())
                        data[method] = time # If duplicate, keep the last one
                    except ValueError:
                        print(f"Warning: Skipping invalid time value in {csv_filepath}: {row[1]} (method: {method})")
                else:
                    print(f"Warning: Skipping incorrectly formatted line in {csv_filepath}: {row}")
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
    return data

if __name__ == '__main__':
    # Read data from CSV files
    cpu_data_file = "cpu_performance_data.csv"
    dcu_data_file = "dcu_performance_data.csv"

    cpu_perf_data = read_performance_data(cpu_data_file)
    dcu_perf_data = read_performance_data(dcu_data_file)

    # Scenario 1: Compare execution times of different CPU optimization methods
    # Define expected CPU method order for chart display
    # MPI method names may include process count, e.g., 'MPI_np4'
    # We need to dynamically extract MPI data from cpu_perf_data
    
    cpu_method_order = [
        'Baseline', 
        'OpenMP', 
        'BlockTiling_bs16', 'BlockTiling_bs32', 'BlockTiling_bs64', 'BlockTiling_bs96', 'BlockTiling_bs128', 'BlockTiling_bs192', 'BlockTiling_bs256',
        'BlockTilingOuterOmp_bs16', 'BlockTilingOuterOmp_bs32', 'BlockTilingOuterOmp_bs64', 'BlockTilingOuterOmp_bs96', 'BlockTilingOuterOmp_bs128', 'BlockTilingOuterOmp_bs192', 'BlockTilingOuterOmp_bs256',
        'BlockTilingImproved_bs16', 'BlockTilingImproved_bs32', 'BlockTilingImproved_bs64', 'BlockTilingImproved_bs96', 'BlockTilingImproved_bs128', 'BlockTilingImproved_bs192', 'BlockTilingImproved_bs256',
        'BlockTilingCacheOpt_bs16', 'BlockTilingCacheOpt_bs32', 'BlockTilingCacheOpt_bs64', 'BlockTilingCacheOpt_bs96', 'BlockTilingCacheOpt_bs128', 'BlockTilingCacheOpt_bs192', 'BlockTilingCacheOpt_bs256',
        'OtherLoopOrder'
    ]
    # Dynamically add MPI data
    mpi_methods_in_data = sorted([m for m in cpu_perf_data if m.startswith("MPI_np")])
    all_cpu_methods_to_plot = cpu_method_order + mpi_methods_in_data
    
    cpu_plot_methods = []
    cpu_plot_times = []
    for method in all_cpu_methods_to_plot:
        if method in cpu_perf_data:
            cpu_plot_methods.append(method)
            cpu_plot_times.append(cpu_perf_data[method])
        else:
            print(f"Info: CPU method '{method}' data not found in {cpu_data_file}.")

    if cpu_plot_methods:
        plot_performance_comparison(cpu_plot_methods, cpu_plot_times, 
                                    title="CPU Matrix Multiplication Performance", save_prefix="cpu_performance")
        
        # Scenario 1.1: CPU optimization methods speedup (relative to CPU Baseline)
        if 'Baseline' in cpu_perf_data and cpu_perf_data['Baseline'] > 0:
            baseline_cpu_time = cpu_perf_data['Baseline']
            optimized_cpu_plot_methods = []
            optimized_cpu_plot_times = []
            # Exclude 'Baseline' from cpu_plot_methods
            for i, method in enumerate(cpu_plot_methods):
                if method != 'Baseline':
                    optimized_cpu_plot_methods.append(method)
                    optimized_cpu_plot_times.append(cpu_plot_times[i])
            
            if optimized_cpu_plot_methods:
                 plot_speedup_comparison(baseline_cpu_time, optimized_cpu_plot_methods, optimized_cpu_plot_times,
                                        title="CPU Optimization Speedup (vs Baseline)", save_prefix="cpu_speedup")
    else:
        print(f"No sufficient CPU performance data found for plotting ({cpu_data_file}).")


    # Scenario 2: Compare CPU baseline with DCU (HIP) execution times
    # Usually comparing CPU baseline from sourcefile_dcu.cpp with HIP kernel/total time
    dcu_comparison_methods_order = ['CPU_Baseline_for_DCU_comparison', 'HIP_Kernel', 'HIP_Total_Incl_Memcpy']
    dcu_plot_methods = []
    dcu_plot_times = []

    for method in dcu_comparison_methods_order:
        if method in dcu_perf_data:
            dcu_plot_methods.append(method)
            dcu_plot_times.append(dcu_perf_data[method])
        else:
            print(f"Info: DCU/HIP method '{method}' data not found in {dcu_data_file}.")
            
    if dcu_plot_methods:
        plot_performance_comparison(dcu_plot_methods, dcu_plot_times,
                                    title="CPU (for DCU) vs HIP Performance", save_prefix="dcu_performance")

        # Scenario 2.1: HIP speedup (relative to its corresponding CPU baseline)
        if 'CPU_Baseline_for_DCU_comparison' in dcu_perf_data and dcu_perf_data['CPU_Baseline_for_DCU_comparison'] > 0:
            baseline_dcu_cpu_time = dcu_perf_data['CPU_Baseline_for_DCU_comparison']
            optimized_dcu_plot_methods = []
            optimized_dcu_plot_times = []
            for i, method in enumerate(dcu_plot_methods):
                if method != 'CPU_Baseline_for_DCU_comparison': # Only compare HIP methods
                    optimized_dcu_plot_methods.append(method)
                    optimized_dcu_plot_times.append(dcu_plot_times[i])
            
            if optimized_dcu_plot_methods:
                plot_speedup_comparison(baseline_dcu_cpu_time, optimized_dcu_plot_methods, optimized_dcu_plot_times,
                                        title="HIP Speedup (vs CPU_Baseline_for_DCU_comparison)", save_prefix="hip_speedup")
    else:
        print(f"No sufficient DCU/HIP performance data found for plotting ({dcu_data_file}).")


    if not cpu_plot_methods and not dcu_plot_methods:
         print("\nNo data loaded from CSV files. Please run C++ programs first to generate data files:")
         print(f" - {cpu_data_file}")
         print(f" - {dcu_data_file}")

    print("\nScript execution completed. Charts have been saved as PNG files.")
    print("Generated files:")
    print(" - cpu_performance_comparison.png")
    print(" - cpu_speedup_speedup.png")
    print(" - dcu_performance_comparison.png (if DCU data available)")
    print(" - hip_speedup_speedup.png (if DCU data available)")
