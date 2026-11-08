#include "bridge.hpp"
using namespace ruv_cv;
namespace ruv_cv { thread_local char error[1024] = {}; }
extern "C" {
const char* cv_last_error(void) { return error; }
const char* cv_version(void) { return CV_VERSION; }
cv_Mat* cv_Mat_new(void) {
    return guard<cv_Mat*>(nullptr, [] { return new cv_Mat{}; });
}
cv_Mat* cv_Mat_new_size_type_scalar(int rows, int cols, int type, cv_Scalar scalar) {
    return guard<cv_Mat*>(nullptr, [&] {
        image_bytes(rows, cols, type);
        return new cv_Mat{cv::Mat(rows, cols, type, cv::Scalar(scalar.val[0], scalar.val[1], scalar.val[2], scalar.val[3]))};
    });
}
cv_Mat* cv_Mat_from_bytes(const unsigned char* data, size_t len, int rows, int cols, int type) {
    return guard<cv_Mat*>(nullptr, [&] {
        const auto bytes = image_bytes(rows, cols, type);
        if (len != bytes || (bytes && !data)) throw std::invalid_argument("Input byte length does not match matrix");
        cv::Mat result(rows, cols, type);
        if (bytes) std::memcpy(result.data, data, bytes);
        return new cv_Mat{std::move(result)};
    });
}
cv_Mat* cv_Mat_from_data(const void* data, int rows, int cols, int type) {
    return guard<cv_Mat*>(nullptr, [&] {
        const auto bytes = image_bytes(rows, cols, type);
        if (bytes && !data) throw std::invalid_argument("NULL input pixels");
        cv::Mat result(rows, cols, type);
        if (bytes) std::memcpy(result.data, data, bytes);
        return new cv_Mat{std::move(result)};
    });
}
void cv_Mat_delete(cv_Mat* value) { status([&] { delete value; }); }
cv_Mat* cv_Mat_clone(const cv_Mat* value) {
    return guard<cv_Mat*>(nullptr, [&] { return new cv_Mat{mat(value).clone()}; });
}
int cv_Mat_copyTo(const cv_Mat* src, cv_Mat* dst) {
    return status([&] { mat(src).copyTo(output(dst)); });
}
int cv_Mat_convertTo(const cv_Mat* src, cv_Mat* dst, int type, double alpha, double beta) {
    return status([&] {
        const auto& source = mat(src); check_finite(alpha); check_finite(beta);
        if (type < 0) type = source.type();
        image_bytes(source.rows, source.cols, CV_MAKETYPE(CV_MAT_DEPTH(type), source.channels()));
        source.convertTo(output(dst), type, alpha, beta);
    });
}
#define CV_MAT_QUERY(Name, Expr) int cv_Mat_##Name(const cv_Mat* value) { return guard<int>(-1, [&] { const auto& m = mat(value); return int(Expr); }); }
CV_MAT_QUERY(rows, m.rows)
CV_MAT_QUERY(cols, m.cols)
CV_MAT_QUERY(type, m.type())
CV_MAT_QUERY(channels, m.channels())
CV_MAT_QUERY(empty, m.empty())
CV_MAT_QUERY(is_continuous, m.isContinuous())
#undef CV_MAT_QUERY
size_t cv_Mat_step(const cv_Mat* value) { return guard<size_t>(SIZE_MAX, [&] { return size_t(mat(value).step[0]); }); }
size_t cv_Mat_total(const cv_Mat* value) { return guard<size_t>(SIZE_MAX, [&] { return mat(value).total(); }); }
size_t cv_Mat_element_size(const cv_Mat* value) { return guard<size_t>(SIZE_MAX, [&] { return mat(value).elemSize(); }); }
unsigned char* cv_Mat_data(const cv_Mat* value) { return guard<unsigned char*>(nullptr, [&] { return mat(value).data; }); }
unsigned char* cv_Mat_row_data(const cv_Mat* value, int row) {
    return guard<unsigned char*>(nullptr, [&] {
        const auto& image = mat(value);
        if (row < 0 || row >= image.rows) throw std::out_of_range("Matrix row out of bounds");
        return const_cast<unsigned char*>(image.ptr(row));
    });
}
int cv_Mat_copy_data(const cv_Mat* value, unsigned char* dst, size_t len) {
    return status([&] {
        const auto& image = mat(value);
        const auto bytes = image_bytes(image.rows, image.cols, image.type());
        if (len < bytes || (bytes && !dst)) throw std::invalid_argument("Output buffer is too small or NULL");
        if (!bytes) return;
        const size_t row_bytes = size_t(image.cols) * image.elemSize();
        for (int row = 0; row < image.rows; ++row) std::memcpy(dst + size_t(row) * row_bytes, image.ptr(row), row_bytes);
    });
}
cv_Mat* cv_Mat_roi(const cv_Mat* value, int x, int y, int width, int height) {
    return guard<cv_Mat*>(nullptr, [&] {
        const auto& source = mat(value);
        if (x < 0 || y < 0 || width < 0 || height < 0 || x > source.cols || y > source.rows || width > source.cols-x || height > source.rows-y)
            throw std::out_of_range("ROI is outside the source matrix");
        return new cv_Mat{source(cv::Rect(x, y, width, height))};
    });
}
}
