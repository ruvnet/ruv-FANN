#include "bridge.hpp"
#ifdef RUV_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
struct cv_cuda_GpuMat { cv::cuda::GpuMat value; };
using namespace ruv_cv;
extern "C" {
cv_cuda_GpuMat* cv_cuda_GpuMat_new(void) {
    return guard<cv_cuda_GpuMat*>(nullptr, [] {
        if (cv::cuda::getCudaEnabledDeviceCount() <= 0) throw std::runtime_error("No CUDA-enabled OpenCV device is available");
        return new cv_cuda_GpuMat{};
    });
}
void cv_cuda_GpuMat_delete(cv_cuda_GpuMat* value) { status([&] { delete value; }); }
int cv_cuda_GpuMat_upload(cv_cuda_GpuMat* gpu, const cv_Mat* host) { return status([&] { require(gpu).value.upload(mat(host)); }); }
int cv_cuda_GpuMat_download(const cv_cuda_GpuMat* gpu, cv_Mat* host) { return status([&] { require(gpu).value.download(output(host)); }); }
int cv_cuda_blur(const cv_cuda_GpuMat* src, cv_cuda_GpuMat* dst, int width, int height) {
    return status([&] {
        const auto& input = require(src).value;
        cv::cuda::createBoxFilter(input.type(), input.type(), kernel(width,height))->apply(input, require(dst).value);
    });
}
int cv_cuda_GaussianBlur(const cv_cuda_GpuMat* src, cv_cuda_GpuMat* dst, int width, int height, double sigma_x, double sigma_y) {
    return status([&] {
        check_finite(sigma_x); check_finite(sigma_y); const auto& input = require(src).value;
        cv::cuda::createGaussianFilter(input.type(), input.type(), kernel(width,height), sigma_x, sigma_y)->apply(input, require(dst).value);
    });
}
#endif
