#ifndef CAFFE_CUDNN_CONV_LAYER_V8_UTILS_HPP_
#define CAFFE_CUDNN_CONV_LAYER_V8_UTILS_HPP_

#ifdef USE_CUDNN
#include <vector>
#include <algorithm>

#include "caffe/util/cudnn.hpp"
#if CUDNN_VERSION_MIN(8, 0, 0) && defined(USE_CUDNN_FRONTEND)
#include "cudnn_frontend.h"

// Note: taken from cudnn_frontend/sample/helpers.h
// Note: taken from cudnn_frontend/sample/conv_sample.cpp

namespace caffe
{

    int64_t checkCudaError(cudaError_t code, const char *expr, const char *file, int line);

    int64_t getCudaComputeCapability();

#define checkCudaErr(...)                                                            \
    do                                                                               \
    {                                                                                \
        int64_t err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    } while (0)

    void generateStrides(const int64_t *dim,
                         int64_t *stride,
                         const int64_t &num_dims,
                         const cudnnTensorFormat_t &filter_format);

    bool isNonDeterministic(cudnnBackendDescriptor_t engine_config);

    // For the descriptors tuple
    enum
    {
        X_TENSOR = 0,
        Y_TENSOR,
        W_TENSOR,
        Z_TENSOR,
        B_TENSOR,
        AFTERADD_TENSOR,
        AFTERBIAS_TENSOR,
        AFTERCONV_TENSOR,
    };

    std::tuple<cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor>
    create_frontend_conv_tensors(int64_t *x_dim,
                                 int64_t *w_dim,
                                 int64_t *y_dim,
                                 cudnnDataType_t dataType,
                                 cudnnDataType_t computeType);

    std::tuple<cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor>
    create_frontend_conv_add_bias_tensors(int64_t *x_dim,
                                          int64_t *w_dim,
                                          int64_t *y_dim,
                                          cudnnDataType_t dataType,
                                          cudnnDataType_t computeType);

    cudnn_frontend::OperationGraph
    create_graph_operation(int64_t *x_dim,
                           int64_t *w_dim,
                           int64_t *y_dim,
                           int64_t *pad,
                           int64_t *dilation,
                           int64_t *stride,
                           cudnnDataType_t dataType,
                           cudnnDataType_t computeType,
                           cudnnHandle_t handle,
                           bool bias);

    cudnn_frontend::ExecutionPlan
    get_execution_plan(cudnn_frontend::OperationGraph &op_graph,
                       cudnnHandle_t handle);

} // namespace caffe

#endif
#endif
#endif // CAFFE_CUDNN_CONV_LAYER_V8_UTILS_HPP_
