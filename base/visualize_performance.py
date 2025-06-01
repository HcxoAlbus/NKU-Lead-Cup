import matplotlib.pyplot as plt
import numpy as np

def plot_performance_comparison(methods, times, title="Performance Comparison"):
    """
    生成并显示不同优化方法的性能对比柱状图。

    参数:
    methods (list of str): 优化方法的名称列表。
    times (list of float): 对应方法的执行时间列表 (例如，秒)。
    title (str): 图表的标题。
    """
    if not methods or not times or len(methods) != len(times):
        print("错误: 方法和时间列表不能为空，且长度必须相同。")
        return

    x_pos = np.arange(len(methods))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_pos, times, align='center', alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    
    plt.xticks(x_pos, methods, rotation=45, ha="right")
    plt.ylabel('Execution Time (seconds)')
    plt.title(title)
    plt.grid(axis='y', linestyle='--')

    # 在柱状图上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(times), f'{yval:.4f}', ha='center', va='bottom')

    plt.tight_layout() # 调整布局以防止标签重叠
    plt.show()

def plot_speedup_comparison(baseline_time, methods, times, title="Speedup Comparison vs Baseline"):
    """
    生成并显示不同优化方法相对于基线的加速比对比柱状图。

    参数:
    baseline_time (float): 基线方法的执行时间。
    methods (list of str): 优化方法的名称列表 (不包括基线)。
    times (list of float): 对应优化方法的执行时间列表。
    title (str): 图表的标题。
    """
    if baseline_time <= 0:
        print("错误: 基线时间必须为正数。")
        return
    if not methods or not times or len(methods) != len(times):
        print("错误: 方法和时间列表不能为空，且长度必须相同。")
        return

    speedups = [baseline_time / t if t > 0 else 0 for t in times] # 计算加速比
    x_pos = np.arange(len(methods))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_pos, speedups, align='center', alpha=0.7, color=['lightcoral', 'lightgreen', 'gold', 'plum'])
    
    plt.xticks(x_pos, methods, rotation=45, ha="right")
    plt.ylabel('Speedup (Baseline Time / Method Time)')
    plt.title(title)
    plt.axhline(1, color='grey', linestyle='--', linewidth=0.8) # 加速比为1的参考线
    plt.grid(axis='y', linestyle='--')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(speedups), f'{yval:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 示例数据：这些数据应从您的性能分析工具 (如 rocprof) 或计时代码中获取
    # 请将这些占位符值替换为您的实际测量结果
    
    # 场景1: 比较不同 CPU 优化方法的执行时间
    cpu_method_names = ['Baseline', 'OpenMP', 'Block Tiling', 'Other (Loop Order)']
    # 假设的执行时间 (秒) - 请替换为实际数据
    cpu_execution_times = [15.5, 4.2, 3.8, 14.9] 
    plot_performance_comparison(cpu_method_names, cpu_execution_times, 
                                title="CPU Matrix Multiplication Performance")

    # 场景2: 比较 CPU 基线与 DCU (HIP) 的执行时间
    # 注意: DCU 时间应包括数据传输时间，或单独分析计算核心时间和总时间
    overall_method_names = ['CPU Baseline', 'HIP (DCU)']
    # 假设的执行时间 (秒) - 请替换为实际数据
    overall_execution_times = [15.5, 0.5] # 假设 HIP 非常快
    plot_performance_comparison(overall_method_names, overall_execution_times,
                                title="CPU vs DCU Matrix Multiplication Performance")

    # 场景3: CPU 优化方法的加速比 (相对于 CPU Baseline)
    # 从 cpu_execution_times 中提取基线时间和优化方法的时间
    if cpu_execution_times and len(cpu_execution_times) > 0:
        baseline_cpu_time = cpu_execution_times[0]
        optimized_cpu_methods = cpu_method_names[1:]
        optimized_cpu_times = cpu_execution_times[1:]
        if optimized_cpu_methods: # 确保有优化方法可比较
             plot_speedup_comparison(baseline_cpu_time, optimized_cpu_methods, optimized_cpu_times,
                                    title="CPU Optimization Speedup vs Baseline")

    # 场景4: MPI 性能随进程数扩展 (示例)
    # 这类图通常是折线图，显示执行时间或加速比如何随进程数变化
    # 此处仍用柱状图简单示意，实际可能需要收集不同进程数下的数据
    mpi_processes = ['MPI (1 proc)', 'MPI (2 procs)', 'MPI (4 procs)', 'MPI (8 procs)']
    # 假设的执行时间 (秒) - 请替换为实际数据
    mpi_times = [16.0, 8.5, 4.8, 2.9] 
    plot_performance_comparison(mpi_processes, mpi_times, title="MPI Performance Scaling (Execution Time)")

    # 也可以绘制 MPI 加速比图
    if mpi_times and len(mpi_times) > 0:
        # 假设单进程 MPI 作为此处的 "基线"
        plot_speedup_comparison(mpi_times[0], mpi_processes[1:], mpi_times[1:],
                                title="MPI Speedup vs Single Process MPI")

    print("性能图已生成 (如果 matplotlib 已配置为显示它们)。")
    print("请将占位符数据替换为您从性能分析中获得的实际测量值。")
    print("您可以通过计时您的 C++ 应用程序或使用工具 (如 rocprof) 获取这些时间。")
    print("例如，rocprof 可以输出 CSV 文件，您可以使用 Python 解析这些文件。")
