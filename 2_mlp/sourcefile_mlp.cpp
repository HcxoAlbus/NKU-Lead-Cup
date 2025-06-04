#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <iomanip> // For std::fixed and std::setprecision
#include <numeric> // For std::iota
#include <random>  // For std::mt19937 and std::shuffle

// 编译文件
// hipcc sourcefile_mlp.cpp -o mlp_full_dcu
// 执行文件
// ./mlp_full_dcu 或者 hipprof ./mlp_full_dcu

// 预定义参数，可根据需求修改
#define INPUT_DIM 10
#define HIDDEN_DIM1 64  // 增加第一个隐藏层大小
#define HIDDEN_DIM2 32  // 增加第二个隐藏层
#define OUTPUT_DIM 1
#define BATCH_SIZE 128  // 减小批次大小，提高训练稳定性
#define EPOCHS 500      // 增加最大训练轮数
#define LEARNING_RATE 5e-4  // 提高初始学习率
#define EARLY_STOP_PATIENCE 15  // 早停耐心值
#define VAL_RATIO 0.15  // 验证集比例

const std::string BANDWIDTH_FILE_PATH = "starlink_bw.json"; // 数据文件路径

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// HIP kernels函数

// C(M,K) = A(M,N) * B(N,K)
__global__ void matmul(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// output = relu(input)
__global__ void relu_forward(const double* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(0.0, input[idx]);
    }
}

// output = leaky_relu(input, alpha=0.01)
__global__ void leaky_relu_forward(const double* input, double* output, int size, double alpha = 0.01) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0.0 ? input[idx] : alpha * input[idx];
    }
}

// output = matrix + bias (bias is row vector, added to each row of matrix)
__global__ void add_bias_forward(const double* matrix, const double* bias, double* output, int M_rows, int N_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M_rows && col < N_cols) {
        output[row * N_cols + col] = matrix[row * N_cols + col] + bias[col];
    }
}

// squared_errors[i] = (pred[i] - target[i])^2
__global__ void compute_mse_loss_elements(const double* pred, const double* target, double* squared_errors, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double diff = pred[idx] - target[idx];
        squared_errors[idx] = diff * diff;
    }
}

// grad_pred = 2/size * (pred - target)
__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        // total_size here is batch_size * output_dim. For MSE, the gradient is averaged.
        grad[idx] = 2.0 * (pred[idx] - target[idx]) / total_size;
    }
}

// delta_in = delta_out * (input_to_relu > 0 ? 1 : 0)
__global__ void compute_relu_backward(const double* delta_out, const double* input_to_relu, double* delta_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta_in[idx] = (input_to_relu[idx] > 0.0) ? delta_out[idx] : 0.0;
    }
}

// delta_in = delta_out * (input_to_leaky_relu > 0 ? 1 : alpha)
__global__ void compute_leaky_relu_backward(const double* delta_out, const double* input_to_relu, double* delta_in, int size, double alpha = 0.01) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta_in[idx] = (input_to_relu[idx] > 0.0) ? delta_out[idx] : alpha * delta_out[idx];
    }
}

// weights -= lr * grad
__global__ void sgd_update(double* weights, const double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

// Transpose matrix: output(N,M) from input(M,N)
__global__ void transpose_matrix(const double* input, double* output, int M_rows, int N_cols) {
    int row_in = blockIdx.y * blockDim.y + threadIdx.y; // Corresponds to M_rows
    int col_in = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to N_cols

    if (row_in < M_rows && col_in < N_cols) {
        output[col_in * M_rows + row_in] = input[row_in * N_cols + col_in];
    }
}

// Sum gradients for biases: bias_grad[j] = sum_i(matrix_grad[i*num_neurons + j])
__global__ void sum_bias_gradients(const double* matrix_grad, double* bias_grad, int batch_size, int num_neurons) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons) {
        double sum = 0.0;
        for (int i = 0; i < batch_size; ++i) {
            sum += matrix_grad[i * num_neurons + neuron_idx];
        }
        bias_grad[neuron_idx] = sum;
    }
}


// 添加缺少的学习率计算函数
double compute_learning_rate(int epoch, double initial_lr) {
    // 学习率衰减策略：每50个epoch衰减10%
    double decay_factor = 0.95;
    int decay_step = 50;
    return initial_lr * pow(decay_factor, epoch / decay_step);
}

// 加载带宽数据
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    std::vector<double> bandwidth_data;
    size_t start_pos = content.find('[');
    size_t end_pos = content.find(']');
    if (start_pos == std::string::npos || end_pos == std::string::npos || start_pos >= end_pos) {
        std::cerr << "Invalid JSON format: missing '[' or ']'" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string numbers_str = content.substr(start_pos + 1, end_pos - start_pos - 1);
    std::stringstream ss(numbers_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            size_t first_digit = item.find_first_not_of(" \t\n\r\f\v");
            size_t last_digit = item.find_last_not_of(" \t\n\r\f\v");
            if (first_digit != std::string::npos && last_digit != std::string::npos) {
                 item = item.substr(first_digit, (last_digit - first_digit + 1));
                 if (!item.empty()) {
                    bandwidth_data.push_back(std::stod(item));
                 }
            }
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid number in JSON: '" << item << "'" << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Number out of range in JSON: '" << item << "'" << std::endl;
        }
    }
    if (bandwidth_data.empty()) {
        std::cerr << "No data loaded from JSON file." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "[INFO] Loaded " << bandwidth_data.size() << " bandwidth records." << std::endl;
    return bandwidth_data;
}

// 创建数据集
void create_dataset(const std::vector<double>& data,
                    std::vector<double>& X, // Flattened: num_samples * INPUT_DIM
                    std::vector<double>& y, // Flattened: num_samples * OUTPUT_DIM (which is 1)
                    int window_size) {      // window_size is INPUT_DIM
    X.clear();
    y.clear();
    if (data.size() <= static_cast<size_t>(window_size)) { // Ensure data.size() > window_size
        std::cerr << "Data size (" << data.size() << ") too small for window size (" << window_size << ")." << std::endl;
        return;
    }
    for (size_t i = 0; i <= data.size() - window_size - 1; ++i) {
        for (int j = 0; j < window_size; ++j) {
            X.push_back(data[i + j]);
        }
        y.push_back(data[i + window_size]);
    }
    std::cout << "[INFO] Created dataset with " << y.size() << " samples." << std::endl;
}

// 数据归一化处理
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
    return;
}

// 数据反归一化处理
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
    // return; // No return needed for void function
}

// Helper to initialize weights and biases
void initialize_weights_biases(double* arr, int size, bool is_bias = false, int fan_in = 0, int fan_out = 0) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    
    if (is_bias) {
        for (int i = 0; i < size; ++i) arr[i] = 0.01; // Small constant for biases
    } else {
        // Xavier/Glorot initialization for weights (uniform distribution)
        double limit = sqrt(6.0 / (fan_in + fan_out));
        std::uniform_real_distribution<> distrib(-limit, limit);
        for (int i = 0; i < size; ++i) {
            arr[i] = distrib(gen);
        }
    }
}


// ----------------------------- Main -------------------------------
int main() {
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // Step 1: 数据准备阶段
    std::cout << "[INFO] Starting Data Preparation..." << std::endl;
    std::vector<double> raw_bandwidth_data = load_json_bandwidth(BANDWIDTH_FILE_PATH);
    
    double min_bw, max_bw;
    std::vector<double> normalized_bw_data = raw_bandwidth_data; // Copy for normalization
    normalize_data(normalized_bw_data, min_bw, max_bw);
    std::cout << "[INFO] Data normalized. Min: " << min_bw << ", Max: " << max_bw << std::endl;

    std::vector<double> h_X_all, h_y_all; // Host data
    create_dataset(normalized_bw_data, h_X_all, h_y_all, INPUT_DIM);

    if (h_y_all.empty()) {
        std::cerr << "[ERROR] Dataset creation failed or resulted in zero samples." << std::endl;
        return 1;
    }

    size_t total_samples = h_y_all.size();
    size_t val_size = static_cast<size_t>(total_samples * VAL_RATIO);
    size_t train_size = static_cast<size_t>(total_samples * 0.7);  // 70% 训练
    size_t test_size = total_samples - train_size - val_size;

    if (train_size == 0 || val_size == 0 || test_size == 0) {
        std::cerr << "[ERROR] Not enough samples for train/val/test split. Total samples: " << total_samples << std::endl;
        return 1;
    }
     if (train_size < BATCH_SIZE) {
        std::cout << "[WARNING] Train size (" << train_size << ") is less than BATCH_SIZE (" << BATCH_SIZE << "). Consider reducing BATCH_SIZE or getting more data." << std::endl;
    }


    std::vector<double> h_X_train(h_X_all.begin(), h_X_all.begin() + train_size * INPUT_DIM);
    std::vector<double> h_y_train(h_y_all.begin(), h_y_all.begin() + train_size * OUTPUT_DIM);
    std::vector<double> h_X_val(h_X_all.begin() + train_size * INPUT_DIM, h_X_all.begin() + (train_size + val_size) * INPUT_DIM);
    std::vector<double> h_y_val(h_y_all.begin() + train_size * OUTPUT_DIM, h_y_all.begin() + (train_size + val_size) * OUTPUT_DIM);
    std::vector<double> h_X_test(h_X_all.begin() + (train_size + val_size) * INPUT_DIM, h_X_all.end());
    std::vector<double> h_y_test(h_y_all.begin() + (train_size + val_size) * OUTPUT_DIM, h_y_all.end());

    std::cout << "[INFO] Dataset split: Train: " << train_size << ", Val: " << val_size << ", Test: " << test_size << std::endl;

    // Device memory allocation
    double *d_X_batch, *d_y_batch;
    double *d_W_ih1, *d_B_h1, *d_W_h1h2, *d_B_h2, *d_W_ho, *d_B_o; // 权重和偏置
    double *d_H1_raw, *d_H1_biased, *d_H1_activated; // 第一个隐藏层
    double *d_H2_raw, *d_H2_biased, *d_H2_activated; // 第二个隐藏层
    double *d_O_raw, *d_O_pred; // 输出层
    double *d_grad_O, *d_grad_W_ho, *d_grad_B_o; // 输出层梯度
    double *d_grad_H2_activated, *d_grad_H2_biased, *d_grad_W_h1h2, *d_grad_B_h2; // 第二隐藏层梯度
    double *d_grad_H1_activated, *d_grad_H1_biased, *d_grad_W_ih1, *d_grad_B_h1; // 第一隐藏层梯度
    double *d_squared_errors; // For MSE calculation
    double *d_H1_activated_T, *d_H2_activated_T, *d_W_ho_T, *d_W_h1h2_T, *d_X_batch_T; // Transposed matrices

    HIP_CHECK(hipMalloc(&d_X_batch, BATCH_SIZE * INPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_y_batch, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));
    
    HIP_CHECK(hipMalloc(&d_W_ih1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B_h1, HIDDEN_DIM1 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W_h1h2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B_h2, HIDDEN_DIM2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W_ho, HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B_o, OUTPUT_DIM * sizeof(double)));

    HIP_CHECK(hipMalloc(&d_H1_raw, BATCH_SIZE * HIDDEN_DIM1 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_H1_biased, BATCH_SIZE * HIDDEN_DIM1 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_H1_activated, BATCH_SIZE * HIDDEN_DIM1 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_H2_raw, BATCH_SIZE * HIDDEN_DIM2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_H2_biased, BATCH_SIZE * HIDDEN_DIM2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_H2_activated, BATCH_SIZE * HIDDEN_DIM2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_O_raw, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_O_pred, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));

    HIP_CHECK(hipMalloc(&d_grad_O, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_W_ho, HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_B_o, OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_H2_activated, BATCH_SIZE * HIDDEN_DIM2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_H2_biased, BATCH_SIZE * HIDDEN_DIM2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_W_h1h2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_B_h2, HIDDEN_DIM2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_H1_activated, BATCH_SIZE * HIDDEN_DIM1 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_H1_biased, BATCH_SIZE * HIDDEN_DIM1 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_W_ih1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_B_h1, HIDDEN_DIM1 * sizeof(double)));

    HIP_CHECK(hipMalloc(&d_squared_errors, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));

    // For transposed matrices needed in backprop
    HIP_CHECK(hipMalloc(&d_H1_activated_T, HIDDEN_DIM1 * BATCH_SIZE * sizeof(double))); // HIDDEN_DIM x BATCH_SIZE
    HIP_CHECK(hipMalloc(&d_H2_activated_T, HIDDEN_DIM2 * BATCH_SIZE * sizeof(double)));       // OUTPUT_DIM x HIDDEN_DIM
    HIP_CHECK(hipMalloc(&d_W_ho_T, OUTPUT_DIM * HIDDEN_DIM2 * sizeof(double)));       // OUTPUT_DIM x HIDDEN_DIM
    HIP_CHECK(hipMalloc(&d_W_h1h2_T, HIDDEN_DIM2 * HIDDEN_DIM1 * sizeof(double)));   // INPUT_DIM x BATCH_SIZE
    HIP_CHECK(hipMalloc(&d_X_batch_T, INPUT_DIM * BATCH_SIZE * sizeof(double)));   // INPUT_DIM x BATCH_SIZE

    // Initialize weights and biases on host
    std::vector<double> h_W_ih1(INPUT_DIM * HIDDEN_DIM1);
    std::vector<double> h_B_h1(HIDDEN_DIM1);
    std::vector<double> h_W_h1h2(HIDDEN_DIM1 * HIDDEN_DIM2);
    std::vector<double> h_B_h2(HIDDEN_DIM2);
    std::vector<double> h_W_ho(HIDDEN_DIM2 * OUTPUT_DIM);
    std::vector<double> h_B_o(OUTPUT_DIM);

    initialize_weights_biases(h_W_ih1.data(), h_W_ih1.size(), false, INPUT_DIM, HIDDEN_DIM1);
    initialize_weights_biases(h_B_h1.data(), h_B_h1.size(), true);
    initialize_weights_biases(h_W_h1h2.data(), h_W_h1h2.size(), false, HIDDEN_DIM1, HIDDEN_DIM2);
    initialize_weights_biases(h_B_h2.data(), h_B_h2.size(), true);
    initialize_weights_biases(h_W_ho.data(), h_W_ho.size(), false, HIDDEN_DIM2, OUTPUT_DIM);
    initialize_weights_biases(h_B_o.data(), h_B_o.size(), true);

    // Copy initial weights and biases to device
    HIP_CHECK(hipMemcpy(d_W_ih1, h_W_ih1.data(), h_W_ih1.size() * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_h1, h_B_h1.data(), h_B_h1.size() * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W_h1h2, h_W_h1h2.data(), h_W_h1h2.size() * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_h2, h_B_h2.data(), h_B_h2.size() * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W_ho, h_W_ho.data(), h_W_ho.size() * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_o, h_B_o.data(), h_B_o.size() * sizeof(double), hipMemcpyHostToDevice));

    std::cout << "[INFO] Device memory allocated and initialized for 2-hidden-layer MLP." << std::endl;
    std::cout << "[INFO] Network architecture: " << INPUT_DIM << " -> " << HIDDEN_DIM1 << " -> " << HIDDEN_DIM2 << " -> " << OUTPUT_DIM << std::endl;
    std::cout << "[INFO] Starting Training..." << std::endl;
    
    dim3 threadsPerBlock(16, 16); // For 2D kernels like matmul
    dim3 threadsPerBlock1D(256);  // For 1D kernels

    // 早停机制变量
    double best_val_loss = std::numeric_limits<double>::max();
    int patience_counter = 0;
    std::vector<double> train_losses, val_losses;

    auto training_start_time = std::chrono::high_resolution_clock::now();

	// 训练MLP网络，包括前向传播、反向传播、梯度下降、参数更新等
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double current_lr = compute_learning_rate(epoch, LEARNING_RATE);
        double total_epoch_loss = 0.0;
        auto epoch_start_time = std::chrono::high_resolution_clock::now();

        // 训练阶段
        std::vector<size_t> train_indices(train_size);
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::shuffle(train_indices.begin(), train_indices.end(), std::mt19937{std::random_device{}()});

        for (size_t i = 0; i < train_size / BATCH_SIZE; ++i) {
            size_t batch_start_idx = i * BATCH_SIZE;
            // Prepare batch data - consider using shuffled indices
            std::vector<double> current_X_batch(BATCH_SIZE * INPUT_DIM);
            std::vector<double> current_y_batch(BATCH_SIZE * OUTPUT_DIM);

            for(size_t k=0; k < BATCH_SIZE; ++k) {
                size_t sample_idx = train_indices[batch_start_idx + k];
                for(int dim=0; dim < INPUT_DIM; ++dim) {
                    current_X_batch[k * INPUT_DIM + dim] = h_X_train[sample_idx * INPUT_DIM + dim];
                }
                for(int dim=0; dim < OUTPUT_DIM; ++dim) {
                    current_y_batch[k * OUTPUT_DIM + dim] = h_y_train[sample_idx * OUTPUT_DIM + dim];
                }
            }
            
            HIP_CHECK(hipMemcpy(d_X_batch, current_X_batch.data(), BATCH_SIZE * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_y_batch, current_y_batch.data(), BATCH_SIZE * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));

            // Step 2: Forward Propagation
            // Layer 1 (Input -> Hidden)
            dim3 matmul_blocks_ih1((HIDDEN_DIM1 + threadsPerBlock.x - 1) / threadsPerBlock.x, (BATCH_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_ih1, threadsPerBlock, 0, 0, d_X_batch, d_W_ih1, d_H1_raw, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM1);
            hipLaunchKernelGGL(add_bias_forward, matmul_blocks_ih1, threadsPerBlock, 0, 0, d_H1_raw, d_B_h1, d_H1_biased, BATCH_SIZE, HIDDEN_DIM1);
            
            dim3 blocks1D_hidden1((BATCH_SIZE * HIDDEN_DIM1 + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
            hipLaunchKernelGGL(leaky_relu_forward, blocks1D_hidden1, threadsPerBlock1D, 0, 0, d_H1_biased, d_H1_activated, BATCH_SIZE * HIDDEN_DIM1);

            // Layer 2 (Hidden1 -> Hidden2)
            dim3 matmul_blocks_h1h2((HIDDEN_DIM2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (BATCH_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_h1h2, threadsPerBlock, 0, 0, d_H1_activated, d_W_h1h2, d_H2_raw, BATCH_SIZE, HIDDEN_DIM1, HIDDEN_DIM2);
            hipLaunchKernelGGL(add_bias_forward, matmul_blocks_h1h2, threadsPerBlock, 0, 0, d_H2_raw, d_B_h2, d_H2_biased, BATCH_SIZE, HIDDEN_DIM2);
            
            dim3 blocks1D_hidden2((BATCH_SIZE * HIDDEN_DIM2 + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
            hipLaunchKernelGGL(leaky_relu_forward, blocks1D_hidden2, threadsPerBlock1D, 0, 0, d_H2_biased, d_H2_activated, BATCH_SIZE * HIDDEN_DIM2);

            // Layer 3 (Hidden2 -> Output)
            dim3 matmul_blocks_ho((OUTPUT_DIM + threadsPerBlock.x - 1) / threadsPerBlock.x, (BATCH_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_ho, threadsPerBlock, 0, 0, d_H2_activated, d_W_ho, d_O_raw, BATCH_SIZE, HIDDEN_DIM2, OUTPUT_DIM);
            hipLaunchKernelGGL(add_bias_forward, matmul_blocks_ho, threadsPerBlock, 0, 0, d_O_raw, d_B_o, d_O_pred, BATCH_SIZE, OUTPUT_DIM);
            
            // Step 3: Error Calculation
            dim3 blocks1D_output((BATCH_SIZE * OUTPUT_DIM + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
            hipLaunchKernelGGL(compute_mse_loss_elements, blocks1D_output, threadsPerBlock1D, 0, 0, d_O_pred, d_y_batch, d_squared_errors, BATCH_SIZE * OUTPUT_DIM);
            
            std::vector<double> h_squared_errors(BATCH_SIZE * OUTPUT_DIM);
            HIP_CHECK(hipMemcpy(h_squared_errors.data(), d_squared_errors, BATCH_SIZE * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
            double batch_loss = 0.0;
            for(double sq_err : h_squared_errors) batch_loss += sq_err;
            batch_loss /= (BATCH_SIZE * OUTPUT_DIM); // Average MSE for the batch
            total_epoch_loss += batch_loss;

            // Step 4: Backward Propagation
            // Output layer gradients
            hipLaunchKernelGGL(compute_output_grad, blocks1D_output, threadsPerBlock1D, 0, 0, d_O_pred, d_y_batch, d_grad_O, BATCH_SIZE * OUTPUT_DIM);

            // Transpose d_H1_activated for d_grad_W_ho calculation
            dim3 transpose_blocks_H1((BATCH_SIZE + threadsPerBlock.x -1) / threadsPerBlock.x, (HIDDEN_DIM1 + threadsPerBlock.y -1)/ threadsPerBlock.y); // M=BS, N=HD
            hipLaunchKernelGGL(transpose_matrix, transpose_blocks_H1, threadsPerBlock, 0, 0, d_H1_activated, d_H1_activated_T, BATCH_SIZE, HIDDEN_DIM1);
            
            dim3 matmul_blocks_grad_Who((OUTPUT_DIM + threadsPerBlock.x -1)/threadsPerBlock.x, (HIDDEN_DIM1 + threadsPerBlock.y-1)/threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_grad_Who, threadsPerBlock, 0, 0, d_H1_activated_T, d_grad_O, d_grad_W_ho, HIDDEN_DIM1, BATCH_SIZE, OUTPUT_DIM);
            
            dim3 blocks1D_bias_O((OUTPUT_DIM + threadsPerBlock1D.x -1)/threadsPerBlock1D.x);
            hipLaunchKernelGGL(sum_bias_gradients, blocks1D_bias_O, threadsPerBlock1D, 0, 0, d_grad_O, d_grad_B_o, BATCH_SIZE, OUTPUT_DIM);

            // 第二隐藏层梯度
            // Transpose d_W_ho for d_grad_H2_activated calculation
            dim3 transpose_blocks_Who((HIDDEN_DIM2 + threadsPerBlock.x -1)/threadsPerBlock.x, (OUTPUT_DIM + threadsPerBlock.y -1)/threadsPerBlock.y); // M=HD, N=OD
            hipLaunchKernelGGL(transpose_matrix, transpose_blocks_Who, threadsPerBlock, 0, 0, d_W_ho, d_W_ho_T, HIDDEN_DIM2, OUTPUT_DIM);

            dim3 matmul_blocks_grad_H2act((HIDDEN_DIM2 + threadsPerBlock.x -1)/threadsPerBlock.x, (BATCH_SIZE + threadsPerBlock.y -1)/threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_grad_H2act, threadsPerBlock, 0, 0, d_grad_O, d_W_ho_T, d_grad_H2_activated, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM2);
            
            hipLaunchKernelGGL(compute_leaky_relu_backward, blocks1D_hidden2, threadsPerBlock1D, 0, 0, d_grad_H2_activated, d_H2_biased, d_grad_H2_biased, BATCH_SIZE * HIDDEN_DIM2);

            // 第二隐藏层权重梯度
            dim3 transpose_blocks_H1h2((HIDDEN_DIM2 + threadsPerBlock.x -1)/threadsPerBlock.x, (HIDDEN_DIM1 + threadsPerBlock.y -1)/threadsPerBlock.y); // M=HD, N=OD
            hipLaunchKernelGGL(transpose_matrix, transpose_blocks_H1h2, threadsPerBlock, 0, 0, d_W_h1h2, d_W_h1h2_T, HIDDEN_DIM2, HIDDEN_DIM1);

            dim3 matmul_blocks_grad_Wh1h2((HIDDEN_DIM2 + threadsPerBlock.x -1)/threadsPerBlock.x, (HIDDEN_DIM1 + threadsPerBlock.y -1)/threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_grad_Wh1h2, threadsPerBlock, 0, 0, d_H1_activated_T, d_grad_H2_biased, d_grad_W_h1h2, HIDDEN_DIM1, BATCH_SIZE, HIDDEN_DIM2);

            dim3 blocks1D_bias_H2((HIDDEN_DIM2 + threadsPerBlock1D.x -1)/threadsPerBlock1D.x);
            hipLaunchKernelGGL(sum_bias_gradients, blocks1D_bias_H2, threadsPerBlock1D, 0, 0, d_grad_H2_biased, d_grad_B_h2, BATCH_SIZE, HIDDEN_DIM2);

            // 第一隐藏层梯度
            dim3 transpose_blocks_Wh1h2((HIDDEN_DIM1 + threadsPerBlock.x -1)/threadsPerBlock.x, (HIDDEN_DIM2 + threadsPerBlock.y -1)/threadsPerBlock.y); // M=HD, N=OD
            hipLaunchKernelGGL(transpose_matrix, transpose_blocks_Wh1h2, threadsPerBlock, 0, 0, d_W_h1h2, d_W_h1h2_T, HIDDEN_DIM1, HIDDEN_DIM2);

            dim3 matmul_blocks_grad_H1act((HIDDEN_DIM1 + threadsPerBlock.x -1)/threadsPerBlock.x, (BATCH_SIZE + threadsPerBlock.y -1)/threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_grad_H1act, threadsPerBlock, 0, 0, d_grad_H2_biased, d_W_h1h2_T, d_grad_H1_activated, BATCH_SIZE, HIDDEN_DIM2, HIDDEN_DIM1);
            
            hipLaunchKernelGGL(compute_leaky_relu_backward, blocks1D_hidden1, threadsPerBlock1D, 0, 0, d_grad_H1_activated, d_H1_biased, d_grad_H1_biased, BATCH_SIZE * HIDDEN_DIM1);

            // 第一隐藏层权重梯度
            dim3 transpose_blocks_X((BATCH_SIZE + threadsPerBlock.x -1)/threadsPerBlock.x, (INPUT_DIM + threadsPerBlock.y -1)/threadsPerBlock.y); // M=BS, N=ID
            hipLaunchKernelGGL(transpose_matrix, transpose_blocks_X, threadsPerBlock, 0, 0, d_X_batch, d_X_batch_T, BATCH_SIZE, INPUT_DIM);

            dim3 matmul_blocks_grad_Wih1((HIDDEN_DIM1 + threadsPerBlock.x -1)/threadsPerBlock.x, (INPUT_DIM + threadsPerBlock.y -1)/threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_grad_Wih1, threadsPerBlock, 0, 0, d_X_batch_T, d_grad_H1_biased, d_grad_W_ih1, INPUT_DIM, BATCH_SIZE, HIDDEN_DIM1);

            dim3 blocks1D_bias_H1((HIDDEN_DIM1 + threadsPerBlock1D.x -1)/threadsPerBlock1D.x);
            hipLaunchKernelGGL(sum_bias_gradients, blocks1D_bias_H1, threadsPerBlock1D, 0, 0, d_grad_H1_biased, d_grad_B_h1, BATCH_SIZE, HIDDEN_DIM1);

            // Step 5: Parameter Update (SGD)
            hipLaunchKernelGGL(sgd_update, dim3((INPUT_DIM * HIDDEN_DIM1 + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x), threadsPerBlock1D, 0, 0, d_W_ih1, d_grad_W_ih1, current_lr, INPUT_DIM * HIDDEN_DIM1);
            hipLaunchKernelGGL(sgd_update, blocks1D_bias_H1, threadsPerBlock1D, 0, 0, d_B_h1, d_grad_B_h1, current_lr, HIDDEN_DIM1);
            hipLaunchKernelGGL(sgd_update, dim3((HIDDEN_DIM1 * HIDDEN_DIM2 + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x), threadsPerBlock1D, 0, 0, d_W_h1h2, d_grad_W_h1h2, current_lr, HIDDEN_DIM1 * HIDDEN_DIM2);
            hipLaunchKernelGGL(sgd_update, blocks1D_bias_H2, threadsPerBlock1D, 0, 0, d_B_h2, d_grad_B_h2, current_lr, HIDDEN_DIM2);
            hipLaunchKernelGGL(sgd_update, dim3((HIDDEN_DIM2 * OUTPUT_DIM + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x), threadsPerBlock1D, 0, 0, d_W_ho, d_grad_W_ho, current_lr, HIDDEN_DIM2 * OUTPUT_DIM);
            hipLaunchKernelGGL(sgd_update, blocks1D_bias_O, threadsPerBlock1D, 0, 0, d_B_o, d_grad_B_o, current_lr, OUTPUT_DIM);
            
            HIP_CHECK(hipDeviceSynchronize());
        }

        // 验证阶段
        double val_loss = 0.0;
        int num_val_batches = (val_size + BATCH_SIZE - 1) / BATCH_SIZE;
        for (int i = 0; i < num_val_batches; ++i) {
            size_t current_batch_start = i * BATCH_SIZE;
            size_t current_batch_actual_size = std::min(static_cast<size_t>(BATCH_SIZE), val_size - current_batch_start);
            if (current_batch_actual_size == 0) continue;

            HIP_CHECK(hipMemcpy(d_X_batch, h_X_val.data() + current_batch_start * INPUT_DIM, current_batch_actual_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_y_batch, h_y_val.data() + current_batch_start * OUTPUT_DIM, current_batch_actual_size * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));

            // 验证前向传播
            dim3 matmul_blocks_ih1_val((HIDDEN_DIM1 + threadsPerBlock.x - 1) / threadsPerBlock.x, (current_batch_actual_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_ih1_val, threadsPerBlock, 0, 0, d_X_batch, d_W_ih1, d_H1_raw, current_batch_actual_size, INPUT_DIM, HIDDEN_DIM1);
            hipLaunchKernelGGL(add_bias_forward, matmul_blocks_ih1_val, threadsPerBlock, 0, 0, d_H1_raw, d_B_h1, d_H1_biased, current_batch_actual_size, HIDDEN_DIM1);
            
            dim3 blocks1D_hidden1_val((current_batch_actual_size * HIDDEN_DIM1 + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
            hipLaunchKernelGGL(leaky_relu_forward, blocks1D_hidden1_val, threadsPerBlock1D, 0, 0, d_H1_biased, d_H1_activated, current_batch_actual_size * HIDDEN_DIM1);

            // 添加第二隐藏层前向传播
            dim3 matmul_blocks_h1h2_val((HIDDEN_DIM2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (current_batch_actual_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_h1h2_val, threadsPerBlock, 0, 0, d_H1_activated, d_W_h1h2, d_H2_raw, current_batch_actual_size, HIDDEN_DIM1, HIDDEN_DIM2);
            hipLaunchKernelGGL(add_bias_forward, matmul_blocks_h1h2_val, threadsPerBlock, 0, 0, d_H2_raw, d_B_h2, d_H2_biased, current_batch_actual_size, HIDDEN_DIM2);
            
            dim3 blocks1D_hidden2_val((current_batch_actual_size * HIDDEN_DIM2 + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
            hipLaunchKernelGGL(leaky_relu_forward, blocks1D_hidden2_val, threadsPerBlock1D, 0, 0, d_H2_biased, d_H2_activated, current_batch_actual_size * HIDDEN_DIM2);

            dim3 matmul_blocks_ho_val((OUTPUT_DIM + threadsPerBlock.x - 1) / threadsPerBlock.x, (current_batch_actual_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
            hipLaunchKernelGGL(matmul, matmul_blocks_ho_val, threadsPerBlock, 0, 0, d_H2_activated, d_W_ho, d_O_raw, current_batch_actual_size, HIDDEN_DIM2, OUTPUT_DIM);
            hipLaunchKernelGGL(add_bias_forward, matmul_blocks_ho_val, threadsPerBlock, 0, 0, d_O_raw, d_B_o, d_O_pred, current_batch_actual_size, OUTPUT_DIM);

            dim3 blocks1D_output_val((current_batch_actual_size * OUTPUT_DIM + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
            hipLaunchKernelGGL(compute_mse_loss_elements, blocks1D_output_val, threadsPerBlock1D, 0, 0, d_O_pred, d_y_batch, d_squared_errors, current_batch_actual_size * OUTPUT_DIM);
            
            std::vector<double> h_squared_errors_val(current_batch_actual_size * OUTPUT_DIM);
            HIP_CHECK(hipMemcpy(h_squared_errors_val.data(), d_squared_errors, current_batch_actual_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
            for(double sq_err : h_squared_errors_val) val_loss += sq_err;
        }
        val_loss /= (val_size * OUTPUT_DIM);

        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        long long epoch_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time).count();
        double avg_epoch_loss = total_epoch_loss / (train_size / BATCH_SIZE);
        
        train_losses.push_back(avg_epoch_loss);
        val_losses.push_back(val_loss);
        
        std::cout << "[Epoch " << std::setw(3) << epoch + 1 << "/" << EPOCHS << "] Train Loss: " << std::fixed << std::setprecision(6) << avg_epoch_loss 
                  << ", Val Loss: " << val_loss << ", LR: " << current_lr << ", Time: " << epoch_duration_ms << " ms" << std::endl;

        // 早停机制
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
        }

        if (patience_counter >= EARLY_STOP_PATIENCE) {
            std::cout << "[INFO] Early stopping triggered at epoch " << epoch + 1 << ". Best validation loss: " << best_val_loss << std::endl;
            break;
        }
    }

    auto training_end_time = std::chrono::high_resolution_clock::now();
    long long total_training_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(training_end_time - training_start_time).count();
    double training_throughput = static_cast<double>(train_size * train_losses.size()) / (total_training_time_ms / 1000.0);
    std::cout << "[INFO] Training finished. Total time: " << total_training_time_ms << " ms." << std::endl;
    std::cout << "[INFO] Training throughput: " << std::fixed << std::setprecision(2) << training_throughput << " samples/sec." << std::endl;

    // 推理阶段 (修复时间测量)
    std::cout << "[INFO] Starting Inference on Test Set..." << std::endl;
    std::vector<double> h_predictions_normalized(test_size * OUTPUT_DIM);
    double total_mse_test = 0.0;
    int num_test_batches = (test_size + BATCH_SIZE -1) / BATCH_SIZE;

    auto inference_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_test_batches; ++i) {
        size_t current_batch_start = i * BATCH_SIZE;
        size_t current_batch_actual_size = std::min(static_cast<size_t>(BATCH_SIZE), test_size - current_batch_start);
        if (current_batch_actual_size == 0) continue;

        HIP_CHECK(hipMemcpy(d_X_batch, h_X_test.data() + current_batch_start * INPUT_DIM, current_batch_actual_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
        
        // 测试前向传播
        dim3 matmul_blocks_ih1_test((HIDDEN_DIM1 + threadsPerBlock.x - 1) / threadsPerBlock.x, (current_batch_actual_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        hipLaunchKernelGGL(matmul, matmul_blocks_ih1_test, threadsPerBlock, 0, 0, d_X_batch, d_W_ih1, d_H1_raw, current_batch_actual_size, INPUT_DIM, HIDDEN_DIM1);
        hipLaunchKernelGGL(add_bias_forward, matmul_blocks_ih1_test, threadsPerBlock, 0, 0, d_H1_raw, d_B_h1, d_H1_biased, current_batch_actual_size, HIDDEN_DIM1);
        
        dim3 blocks1D_hidden1_test((current_batch_actual_size * HIDDEN_DIM1 + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
        hipLaunchKernelGGL(leaky_relu_forward, blocks1D_hidden1_test, threadsPerBlock1D, 0, 0, d_H1_biased, d_H1_activated, current_batch_actual_size * HIDDEN_DIM1);

        dim3 matmul_blocks_h1h2_test((HIDDEN_DIM2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (current_batch_actual_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        hipLaunchKernelGGL(matmul, matmul_blocks_h1h2_test, threadsPerBlock, 0, 0, d_H1_activated, d_W_h1h2, d_H2_raw, current_batch_actual_size, HIDDEN_DIM1, HIDDEN_DIM2);
        hipLaunchKernelGGL(add_bias_forward, matmul_blocks_h1h2_test, threadsPerBlock, 0, 0, d_H2_raw, d_B_h2, d_H2_biased, current_batch_actual_size, HIDDEN_DIM2);
        
        dim3 blocks1D_hidden2_test((current_batch_actual_size * HIDDEN_DIM2 + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
        hipLaunchKernelGGL(leaky_relu_forward, blocks1D_hidden2_test, threadsPerBlock1D, 0, 0, d_H2_biased, d_H2_activated, current_batch_actual_size * HIDDEN_DIM2);

        dim3 matmul_blocks_ho_test((OUTPUT_DIM + threadsPerBlock.x - 1) / threadsPerBlock.x, (current_batch_actual_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        hipLaunchKernelGGL(matmul, matmul_blocks_ho_test, threadsPerBlock, 0, 0, d_H2_activated, d_W_ho, d_O_raw, current_batch_actual_size, HIDDEN_DIM2, OUTPUT_DIM);
        hipLaunchKernelGGL(add_bias_forward, matmul_blocks_ho_test, threadsPerBlock, 0, 0, d_O_raw, d_B_o, d_O_pred, current_batch_actual_size, OUTPUT_DIM);

        HIP_CHECK(hipMemcpy(h_predictions_normalized.data() + current_batch_start * OUTPUT_DIM, d_O_pred, current_batch_actual_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
    
        // 计算MSE
        for (size_t j = 0; j < current_batch_actual_size; ++j) {
            double pred_val = h_predictions_normalized[current_batch_start * OUTPUT_DIM + j];
            double actual_val = h_y_test[current_batch_start * OUTPUT_DIM + j];
            double diff = pred_val - actual_val;
            total_mse_test += diff * diff;
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto inference_end_time = std::chrono::high_resolution_clock::now();
    long long total_inference_time_us = std::chrono::duration_cast<std::chrono::microseconds>(inference_end_time - inference_start_time).count();
    // 修复除零问题，并使用微秒进行更精确的吞吐量计算
    double inference_throughput = total_inference_time_us > 0 ? (static_cast<double>(test_size) * 1000000.0 / total_inference_time_us) : 0.0;


    total_mse_test /= (test_size * OUTPUT_DIM);
    // 以浮点数形式输出毫秒，提高精度
    std::cout << "[INFO] Inference finished. Total time: " << std::fixed << std::setprecision(3) << (total_inference_time_us / 1000.0) << " ms." << std::endl;
    std::cout << "[INFO] Inference throughput: " << std::fixed << std::setprecision(2) << inference_throughput << " samples/sec." << std::endl;
    std::cout << "[INFO] Mean Squared Error (MSE) on (normalized) Test Set: " << std::fixed << std::setprecision(8) << total_mse_test << std::endl;

    // Denormalize predictions and actual test values for comparison
    std::vector<double> h_predictions_denormalized = h_predictions_normalized;
    denormalize_data(h_predictions_denormalized, min_bw, max_bw);
    
    std::vector<double> h_y_test_denormalized = h_y_test; // Copy before denormalizing
    denormalize_data(h_y_test_denormalized, min_bw, max_bw);

    std::cout << "[INFO] Sample Predictions (denormalized):" << std::endl;
    for (size_t i = 0; i < std::min(static_cast<size_t>(10), test_size); ++i) {
        std::cout << "  Sample " << std::setw(2) << i << ": Predicted: " << std::fixed << std::setprecision(2) << h_predictions_denormalized[i]
                  << ", Actual: " << h_y_test_denormalized[i] << std::endl;
    }
    
    // Calculate MSE on denormalized data for a more interpretable error
    double total_mse_denormalized = 0.0;
    for(size_t i=0; i < test_size; ++i) {
        double diff = h_predictions_denormalized[i] - h_y_test_denormalized[i];
        total_mse_denormalized += diff * diff;
    }
    total_mse_denormalized /= (test_size * OUTPUT_DIM);
    std::cout << "[INFO] Mean Squared Error (MSE) on (denormalized) Test Set: " << std::fixed << std::setprecision(2) << total_mse_denormalized << std::endl;


    // Cleanup
    std::cout << "[INFO] Cleaning up device memory..." << std::endl;
    HIP_CHECK(hipFree(d_X_batch)); HIP_CHECK(hipFree(d_y_batch));
    HIP_CHECK(hipFree(d_W_ih1)); HIP_CHECK(hipFree(d_B_h1));
    HIP_CHECK(hipFree(d_W_h1h2)); HIP_CHECK(hipFree(d_B_h2));
    HIP_CHECK(hipFree(d_W_ho)); HIP_CHECK(hipFree(d_B_o));
    HIP_CHECK(hipFree(d_H1_raw)); HIP_CHECK(hipFree(d_H1_biased)); HIP_CHECK(hipFree(d_H1_activated));
    HIP_CHECK(hipFree(d_H2_raw)); HIP_CHECK(hipFree(d_H2_biased)); HIP_CHECK(hipFree(d_H2_activated));
    HIP_CHECK(hipFree(d_O_raw)); HIP_CHECK(hipFree(d_O_pred));
    HIP_CHECK(hipFree(d_grad_O)); HIP_CHECK(hipFree(d_grad_W_ho)); HIP_CHECK(hipFree(d_grad_B_o));
    HIP_CHECK(hipFree(d_grad_H2_activated)); HIP_CHECK(hipFree(d_grad_H2_biased));
    HIP_CHECK(hipFree(d_grad_W_h1h2)); HIP_CHECK(hipFree(d_grad_B_h2));
    HIP_CHECK(hipFree(d_grad_H1_activated)); HIP_CHECK(hipFree(d_grad_H1_biased));
    HIP_CHECK(hipFree(d_grad_W_ih1)); HIP_CHECK(hipFree(d_grad_B_h1));
    HIP_CHECK(hipFree(d_squared_errors));
    HIP_CHECK(hipFree(d_H1_activated_T)); HIP_CHECK(hipFree(d_H2_activated_T));
    HIP_CHECK(hipFree(d_W_ho_T)); HIP_CHECK(hipFree(d_W_h1h2_T)); HIP_CHECK(hipFree(d_X_batch_T));
    std::cout << "[INFO] Device memory freed." << std::endl;

    auto total_end_time = std::chrono::high_resolution_clock::now();
    long long full_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count();
    std::cout << "[INFO] Total execution time: " << full_duration_ms << " ms." << std::endl;
    
    return 0;
}
