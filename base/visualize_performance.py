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
    生成并保存不同优化方法的性能比较柱状图。

    参数:
    methods (list of str): 优化方法名称列表。
    times (list of float): 执行时间列表（单位：秒）。
    title (str): 图表标题。
    save_prefix (str): 保存文件的前缀。
    """
    if not methods or not times or len(methods) != len(times):
        print("错误：方法和时间列表不能为空，且长度必须相同。")
        return

    x_pos = np.arange(len(methods))

    plt.figure(figsize=(15, 8)) # 增大图像尺寸以容纳更多数据
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))  # 使用颜色映射生成足够的颜色
    bars = plt.bar(x_pos, times, align='center', alpha=0.7, color=colors)
    
    plt.xticks(x_pos, methods, rotation=45, ha="right")
    plt.ylabel('执行时间（秒）')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱状图上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(times), f'{yval:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout() # 调整布局以防止标签重叠
    
    # 保存图像
    filename = f"{save_prefix}_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"性能比较图已保存为：{filename}")
    plt.close()  # 关闭图像以释放内存

def plot_speedup_comparison(baseline_time, methods, times, title="Speedup Comparison (vs Baseline)", save_prefix="cpu_speedup"):
    """
    生成并保存相对于基线的加速比比较柱状图。

    参数:
    baseline_time (float): 基线方法的执行时间。
    methods (list of str): 优化方法名称列表（不包括基线）。
    times (list of float): 对应优化方法的执行时间列表。
    title (str): 图表标题。
    save_prefix (str): 保存文件的前缀。
    """
    if baseline_time <= 0:
        print("错误：基线时间必须为正数。")
        return
    if not methods or not times or len(methods) != len(times):
        print("错误：方法和时间列表不能为空，且长度必须相同。")
        return

    speedups = [baseline_time / t if t > 0 else 0 for t in times] 
    x_pos = np.arange(len(methods))

    plt.figure(figsize=(15, 8)) # 增大图像尺寸
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))  # 使用不同的颜色映射
    bars = plt.bar(x_pos, speedups, align='center', alpha=0.7, color=colors)
    
    plt.xticks(x_pos, methods, rotation=45, ha="right")
    plt.ylabel('加速比（基线时间 / 方法时间）')
    plt.title(title)
    plt.axhline(1, color='red', linestyle='--', linewidth=1.0, alpha=0.8, label='无加速参考线（1x）') # 加速比=1的参考线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(speedups), f'{yval:.2f}x', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f"{save_prefix}_speedup.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"加速比比较图已保存为：{filename}")
    plt.close()  # 关闭图像以释放内存

def read_performance_data(csv_filepath):
    """
    从指定的 CSV 文件读取性能数据。
    返回一个字典，键为方法名称，值为最新的执行时间。
    """
    data = {}
    if not os.path.exists(csv_filepath):
        print(f"警告：未找到 CSV 文件：{csv_filepath}")
        return data
        
    try:
        with open(csv_filepath, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader, None) # 跳过表头
            if header is None or [h.strip() for h in header] != ['Method', 'Time']:
                print(f"警告：{csv_filepath} 的表头格式不正确。预期格式为 'Method,Time'")
                # 尝试继续读取，假设为两列格式
            
            for row in reader:
                if len(row) == 2:
                    method = row[0].strip()
                    try:
                        time = float(row[1].strip())
                        data[method] = time # 如果有重复，保留最后一个
                    except ValueError:
                        print(f"警告：跳过 {csv_filepath} 中无效的时间值：{row[1]}（方法：{method}）")
                else:
                    print(f"警告：跳过 {csv_filepath} 中格式不正确的行：{row}")
    except Exception as e:
        print(f"读取 CSV 文件 {csv_filepath} 时发生错误：{e}")
    return data

if __name__ == '__main__':
    # 从 CSV 文件读取数据
    cpu_data_file = "cpu_performance_data.csv"
    dcu_data_file = "dcu_performance_data.csv"

    cpu_perf_data = read_performance_data(cpu_data_file)
    dcu_perf_data = read_performance_data(dcu_data_file)

    # 场景 1：比较不同 CPU 优化方法的执行时间
    # 定义图表显示的预期 CPU 方法顺序
    # MPI 方法名称可能包含进程数，例如 'MPI_np4'
    # 我们需要动态从 cpu_perf_data 中提取 MPI 数据
    
    cpu_method_order = [
        'Baseline', 
        'OpenMP', 
        'BlockTiling_bs16', 'BlockTiling_bs32', 'BlockTiling_bs64', 'BlockTiling_bs96', 'BlockTiling_bs128', 'BlockTiling_bs192', 'BlockTiling_bs256',
        'BlockTilingOuterOmp_bs16', 'BlockTilingOuterOmp_bs32', 'BlockTilingOuterOmp_bs64', 'BlockTilingOuterOmp_bs96', 'BlockTilingOuterOmp_bs128', 'BlockTilingOuterOmp_bs192', 'BlockTilingOuterOmp_bs256',
        'BlockTilingImproved_bs16', 'BlockTilingImproved_bs32', 'BlockTilingImproved_bs64', 'BlockTilingImproved_bs96', 'BlockTilingImproved_bs128', 'BlockTilingImproved_bs192', 'BlockTilingImproved_bs256',
        'BlockTilingCacheOpt_bs16', 'BlockTilingCacheOpt_bs32', 'BlockTilingCacheOpt_bs64', 'BlockTilingCacheOpt_bs96', 'BlockTilingCacheOpt_bs128', 'BlockTilingCacheOpt_bs192', 'BlockTilingCacheOpt_bs256',
        'OtherLoopOrder'
    ]
    # 动态添加 MPI 数据
    mpi_methods_in_data = sorted([m for m in cpu_perf_data if m.startswith("MPI_np")])
    all_cpu_methods_to_plot = cpu_method_order + mpi_methods_in_data
    
    cpu_plot_methods = []
    cpu_plot_times = []
    for method in all_cpu_methods_to_plot:
        if method in cpu_perf_data:
            cpu_plot_methods.append(method)
            cpu_plot_times.append(cpu_perf_data[method])
        else:
            print(f"信息：在 {cpu_data_file} 中未找到 CPU 方法 '{method}' 的数据。")

    if cpu_plot_methods:
        plot_performance_comparison(cpu_plot_methods, cpu_plot_times, 
                                    title="CPU 矩阵乘法性能", save_prefix="cpu_performance")
        
        # 场景 1.1：CPU 优化方法的加速比（相对于 CPU 基线）
        if 'Baseline' in cpu_perf_data and cpu_perf_data['Baseline'] > 0:
            baseline_cpu_time = cpu_perf_data['Baseline']
            optimized_cpu_plot_methods = []
            optimized_cpu_plot_times = []
            # 从 cpu_plot_methods 中排除 'Baseline'
            for i, method in enumerate(cpu_plot_methods):
                if method != 'Baseline':
                    optimized_cpu_plot_methods.append(method)
                    optimized_cpu_plot_times.append(cpu_plot_times[i])
            
            if optimized_cpu_plot_methods:
                 plot_speedup_comparison(baseline_cpu_time, optimized_cpu_plot_methods, optimized_cpu_plot_times,
                                        title="CPU 优化加速比（相对于基线）", save_prefix="cpu_speedup")
    else:
        print(f"未找到足够的 CPU 性能数据用于绘图（{cpu_data_file}）。")


    # 场景 2：比较 CPU 基线与 DCU（HIP）执行时间
    # 通常比较 sourcefile_dcu.cpp 中的 CPU 基线与 HIP 内核/总时间
    dcu_comparison_methods_order = ['CPU_Baseline_for_DCU_comparison', 'HIP_Kernel', 'HIP_Total_Incl_Memcpy']
    dcu_plot_methods = []
    dcu_plot_times = []

    for method in dcu_comparison_methods_order:
        if method in dcu_perf_data:
            dcu_plot_methods.append(method)
            dcu_plot_times.append(dcu_perf_data[method])
        else:
            print(f"信息：在 {dcu_data_file} 中未找到 DCU/HIP 方法 '{method}' 的数据。")
            
    if dcu_plot_methods:
        plot_performance_comparison(dcu_plot_methods, dcu_plot_times,
                                    title="CPU（用于 DCU 比较）与 HIP 性能", save_prefix="dcu_performance")

        # 场景 2.1：HIP 加速比（相对于其对应的 CPU 基线）
        if 'CPU_Baseline_for_DCU_comparison' in dcu_perf_data and dcu_perf_data['CPU_Baseline_for_DCU_comparison'] > 0:
            baseline_dcu_cpu_time = dcu_perf_data['CPU_Baseline_for_DCU_comparison']
            optimized_dcu_plot_methods = []
            optimized_dcu_plot_times = []
            for i, method in enumerate(dcu_plot_methods):
                if method != 'CPU_Baseline_for_DCU_comparison': # 仅比较 HIP 方法
                    optimized_dcu_plot_methods.append(method)
                    optimized_dcu_plot_times.append(dcu_plot_times[i])
            
            if optimized_dcu_plot_methods:
                plot_speedup_comparison(baseline_dcu_cpu_time, optimized_dcu_plot_methods, optimized_dcu_plot_times,
                                        title="HIP 加速比（相对于 CPU_Baseline_for_DCU_comparison）", save_prefix="hip_speedup")
    else:
        print(f"未找到足够的 DCU/HIP 性能数据用于绘图（{dcu_data_file}）。")


    if not cpu_plot_methods and not dcu_plot_methods:
         print("\n未从 CSV 文件加载数据。请先运行 C++ 程序以生成数据文件：")
         print(f" - {cpu_data_file}")
         print(f" - {dcu_data_file}")

    print("\n脚本执行完成。图表已保存为 PNG 文件。")
    print("生成的文件：")
    print(" - cpu_performance_comparison.png")
    print(" - cpu_speedup_speedup.png")
    print(" - dcu_performance_comparison.png（如果有 DCU 数据）")
    print(" - hip_speedup_speedup.png（如果有 DCU 数据）")
