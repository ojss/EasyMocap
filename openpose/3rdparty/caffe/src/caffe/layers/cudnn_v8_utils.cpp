#ifdef USE_CUDNN

#include "caffe/layers/cudnn_v8_utils.hpp"

#if CUDNN_VERSION_MIN(8, 0, 0) && defined(USE_CUDNN_FRONTEND)
namespace caffe
{

    int64_t checkCudaError(cudaError_t code,
                           const char *expr,
                           const char *file,
                           int line)
    {
        if (code)
        {
            printf("CUDA error at %s:%d, code=%d (%s) in '%s'",
                   file, line, (int)code, cudaGetErrorString(code), expr);
            return 1;
        }
        return 0;
    }

    int64_t getCudaComputeCapability()
    {
        cudaDeviceProp prop;
        int device;
        checkCudaErr(cudaGetDevice(&device));
        checkCudaErr(cudaGetDeviceProperties(&prop, device));
        return prop.major * 10 + prop.minor;
    }

    // Note: taken from udnn_frontend/sample/helpers.cpp
    void generateStrides(const int64_t *dim,
                         int64_t *stride,
                         const int64_t &num_dims,
                         const cudnnTensorFormat_t &filter_format)
    {
        // For INT8x4 and INT8x32 we still compute standard strides here to input
        // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
        if (filter_format == CUDNN_TENSOR_NCHW)
        {
            stride[num_dims - 1] = 1;
            for (int64_t d = num_dims - 2; d >= 0; d--)
                stride[d] = stride[d + 1] * dim[d + 1];
        }
        else
        {
            // Here we assume that the format is CUDNN_TENSOR_NHWC
            stride[1] = 1;
            stride[num_dims - 1] = stride[1] * dim[1];
            for (int64_t d = num_dims - 2; d >= 2; d--)
                stride[d] = stride[d + 1] * dim[d + 1];
            stride[0] = stride[2] * dim[2];
        }
    }

    bool isNonDeterministic(cudnnBackendDescriptor_t engine_config)
    {
        return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(engine_config);
    }

    // Note: taken from cudnn_frontend/sample/conv_sample.cpp
    std::tuple<cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor>
    create_frontend_conv_tensors(
        int64_t *x_dim,
        int64_t *w_dim,
        int64_t *y_dim,
        cudnnDataType_t dataType,
        cudnnDataType_t computeType)
    {
        int alignment = (getCudaComputeCapability() >= 80) ? 16 : 4;

        int64_t x_stride[4];
        int64_t y_stride[4];
        int64_t w_stride[4];
        generateStrides(x_dim, x_stride, 4, CUDNN_TENSOR_NCHW);
        generateStrides(y_dim, y_stride, 4, CUDNN_TENSOR_NCHW);
        generateStrides(w_dim, w_stride, 4, CUDNN_TENSOR_NCHW);
        return std::tuple<cudnn_frontend::Tensor,
                          cudnn_frontend::Tensor,
                          cudnn_frontend::Tensor>(
            cudnn_frontend::TensorBuilder()
                .setDim(4, x_dim)
                .setStride(4, x_stride)
                .setId('x')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, y_stride)
                .setId('y')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, w_dim)
                .setStride(4, w_stride)
                .setId('w')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build());
    }

    std::tuple<cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor,
               cudnn_frontend::Tensor>
    create_frontend_conv_add_bias_tensors(
        int64_t *x_dim,
        int64_t *w_dim,
        int64_t *y_dim,
        cudnnDataType_t dataType,
        cudnnDataType_t computeType)
    {
        int alignment = (getCudaComputeCapability() >= 80) ? 16 : 4;

        int64_t x_stride[4];
        int64_t y_stride[4];
        int64_t w_stride[4];
        int64_t b_dim[4] = {1, y_dim[1], 1, 1};
        int64_t b_stride[4];
        generateStrides(x_dim, x_stride, 4, CUDNN_TENSOR_NCHW);
        generateStrides(y_dim, y_stride, 4, CUDNN_TENSOR_NCHW);
        generateStrides(w_dim, w_stride, 4, CUDNN_TENSOR_NCHW);
        generateStrides(b_dim, b_stride, 4, CUDNN_TENSOR_NCHW);
        return std::tuple<cudnn_frontend::Tensor,
                          cudnn_frontend::Tensor,
                          cudnn_frontend::Tensor,
                          cudnn_frontend::Tensor,
                          cudnn_frontend::Tensor,
                          cudnn_frontend::Tensor,
                          cudnn_frontend::Tensor,
                          cudnn_frontend::Tensor>(
            cudnn_frontend::TensorBuilder()
                .setDim(4, x_dim)
                .setStride(4, x_stride)
                .setId('x')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, y_stride)
                .setId('y')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, w_dim)
                .setStride(4, w_stride)
                .setId('w')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, y_stride)
                .setId('z')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, b_dim)
                .setStride(4, b_stride)
                .setId('b')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, y_stride)
                .setId('A') // after add
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(computeType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, y_stride)
                .setVirtual()
                .setId('B') // after bias
                .setAlignment(alignment)
                .setDataType(computeType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, y_stride)
                .setVirtual()
                .setId('C') // after conv
                .setAlignment(alignment)
                .setDataType(computeType)
                .build());
    }

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
                           bool bias)
    {
        // Define the convolution problem
        // Note: In legacy API, convolution operation requires
        // cudnn::setConvolutionDesc(...) which uses
        // `CUDNN_CROSS_CORRELATION` instead of `CUDNN_CONVOLUTION`.
        int conv_dim = 2;
        cudnn_frontend::ConvDesc convDesc = cudnn_frontend::ConvDescBuilder()
                                                .setComputeType(computeType)
                                                .setMathMode(CUDNN_CROSS_CORRELATION)
                                                .setSpatialDimCount(conv_dim)
                                                .setSpatialStride(conv_dim, stride)
                                                .setPrePadding(conv_dim, pad)
                                                .setPostPadding(conv_dim, pad)
                                                .setDilation(conv_dim, dilation)
                                                .build();
        // Define the add operation
        cudnn_frontend::PointwiseDesc addDesc = cudnn_frontend::PointWiseDescBuilder()
                                                    .setMode(CUDNN_POINTWISE_ADD)
                                                    .setMathPrecision(computeType)
                                                    .build();

        if (bias)
        {
            auto tensors = create_frontend_conv_add_bias_tensors(
                x_dim, w_dim, y_dim, dataType, computeType);
            // Create a convolution Node, doesnt have alpha2
            auto conv_op = cudnn_frontend::OperationBuilder(
                               CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                               .setxDesc(std::get<X_TENSOR>(tensors))
                               .setwDesc(std::get<W_TENSOR>(tensors))
                               .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
                               .setcDesc(convDesc)
                               .setAlpha(1.0f)
                               .setBeta(0.0f)
                               .build();
            // Create a dummy add Node.
            auto add_op = cudnn_frontend::OperationBuilder(
                              CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                              .setxDesc(conv_op.getOutputTensor())
                              .setbDesc(std::get<Z_TENSOR>(tensors))
                              .setyDesc(std::get<AFTERADD_TENSOR>(tensors))
                              .setpwDesc(addDesc)
                              .setAlpha(1.0f)
                              .setAlpha2(0.0f)
                              .build();
            // Create a Bias Node.
            auto bias_op = cudnn_frontend::OperationBuilder(
                               CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(add_op.getOutputTensor())
                               .setbDesc(std::get<B_TENSOR>(tensors))
                               .setyDesc(std::get<Y_TENSOR>(tensors))
                               .setpwDesc(addDesc)
                               .setAlpha(1.0f)
                               .setAlpha2(1.0f)
                               .build();
            // Create an Operation Graph.
            const cudnn_frontend::Operation *ops[] = {&conv_op, &add_op, &bias_op};
            return cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle)
                .setOperationGraph(3, ops)
                .build();
        }
        else
        {
            auto tensors = create_frontend_conv_tensors(
                x_dim, w_dim, y_dim, dataType, computeType);
            // Create a convolution Node, doesnt have alpha2
            auto conv_op = cudnn_frontend::OperationBuilder(
                               CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                               .setxDesc(std::get<X_TENSOR>(tensors))
                               .setwDesc(std::get<W_TENSOR>(tensors))
                               .setyDesc(std::get<Y_TENSOR>(tensors))
                               .setcDesc(convDesc)
                               .setAlpha(1.0f)
                               .setBeta(0.0f)
                               .build();
            // Create an Operation Graph.
            const cudnn_frontend::Operation *ops[] = {&conv_op};
            return cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle)
                .setOperationGraph(1, ops)
                .build();
        }
    }

    cudnn_frontend::ExecutionPlan
    get_execution_plan(cudnn_frontend::OperationGraph &op_graph,
                       cudnnHandle_t handle)
    {
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"},
            op_graph, isNonDeterministic, filtered_configs);
        bool plan_found = false;
        for (auto &filtered_config : filtered_configs)
        {
            try
            {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(filtered_config, op_graph.getTag())
                                .build();
                plan_found = true;
                return plan;
            }
            catch (cudnn_frontend::cudnnException &e)
            {
                std::cout << "cudnnException " << e.what() << std::endl;
                continue;
            }
        }
        cudnn_frontend::throw_if([plan_found]()
                                 { return (!plan_found); },
                                 "No plan found for cudnn layer...",
                                 CUDNN_STATUS_EXECUTION_FAILED);
    }

} // namespace caffe
#endif
#endif
