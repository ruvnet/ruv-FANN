#ifndef RUV_OPENCV_BRIDGE_HPP
#define RUV_OPENCV_BRIDGE_HPP
#include "wrapper.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ml.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <climits>
#include <stdexcept>
#include <vector>

struct cv_Mat { cv::Mat value; };
struct cv_VideoCapture { cv::VideoCapture value; };
struct cv_SIFT { cv::Ptr<cv::SIFT> value; };
struct cv_CascadeClassifier { cv::CascadeClassifier value; };
struct cv_ml_SVM { cv::Ptr<cv::ml::SVM> value; };
struct cv_KeyPoint_vector { std::vector<cv::KeyPoint> value; };
struct cv_Rect_vector { std::vector<cv::Rect> value; };
struct cv_Point2f_vector { std::vector<cv::Point2f> value; };
struct cv_Point3f_vector { std::vector<cv::Point3f> value; };
struct cv_Point2f_vector_vector { std::vector<std::vector<cv::Point2f>> value; };
struct cv_Point3f_vector_vector { std::vector<std::vector<cv::Point3f>> value; };

namespace ruv_cv {
extern thread_local char error[1024];
inline void set_error(const char* message) noexcept {
    std::snprintf(error, sizeof(error), "%s", message);
}
template<class R, class F> R guard(R failure, F work) noexcept {
    error[0] = '\0';
    try { return work(); }
    catch (const cv::Exception& e) { set_error(e.what()); }
    catch (const std::exception& e) { set_error(e.what()); }
    catch (...) { set_error("Unknown C++ exception"); }
    return failure;
}
template<class F> int status(F work) noexcept {
    return guard<int>(-1, [&] { work(); return 0; });
}
template<class T> T& require(T* ptr) {
    if (!ptr) throw std::invalid_argument("NULL handle or output pointer");
    return *ptr;
}
inline const char* text(const char* ptr) {
    if (!ptr || !*ptr) throw std::invalid_argument("Empty or NULL string");
    return ptr;
}
constexpr size_t max_bytes = size_t(1) << 30;
inline size_t image_bytes(int rows, int cols, int type) {
    if (rows < 0 || cols < 0 || type < 0 || (type & ~CV_MAT_TYPE_MASK))
        throw std::invalid_argument("Invalid dimensions or OpenCV matrix type");
    const size_t element = CV_ELEM_SIZE(type);
    if (cols && size_t(rows) > max_bytes / element / size_t(cols))
        throw std::length_error("Matrix exceeds 1 GiB bridge allocation limit");
    return size_t(rows) * size_t(cols) * element;
}
inline void bounded(const cv::Mat& value) {
    if (value.dims > 2) throw std::invalid_argument("Only two-dimensional matrices are supported");
    image_bytes(value.rows, value.cols, value.type());
}
inline cv::Mat& mat(cv_Mat* value) { return require(value).value; }
inline const cv::Mat& mat(const cv_Mat* value) { return require(value).value; }
inline cv::Mat& output(cv_Mat* value) { return require(value).value; }
inline cv::Size kernel(int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("Kernel dimensions must be positive");
    return {width, height};
}
inline void check_finite(double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Parameter must be finite");
}
inline void gui() {
#if defined(__linux__)
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
        throw std::runtime_error("OpenCV GUI requires a running display");
#endif
}
}
#endif
