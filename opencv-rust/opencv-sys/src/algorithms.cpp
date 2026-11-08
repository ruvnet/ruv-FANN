#include "bridge.hpp"
using namespace ruv_cv;
extern "C" {
int cv_blur(const cv_Mat* src, cv_Mat* dst, int width, int height) {
    return status([&] { cv::blur(mat(src), output(dst), kernel(width, height)); });
}
int cv_GaussianBlur(const cv_Mat* src, cv_Mat* dst, int width, int height, double sigma_x, double sigma_y) {
    return status([&] { check_finite(sigma_x); check_finite(sigma_y); cv::GaussianBlur(mat(src), output(dst), kernel(width, height), sigma_x, sigma_y); });
}
int cv_Canny(const cv_Mat* src, cv_Mat* dst, double threshold1, double threshold2) {
    return status([&] { check_finite(threshold1); check_finite(threshold2); cv::Canny(mat(src), output(dst), threshold1, threshold2); });
}
int cv_resize(const cv_Mat* src, cv_Mat* dst, int width, int height, double fx, double fy, int interpolation) {
    return status([&] {
        const auto& source = mat(src); check_finite(fx); check_finite(fy);
        if (width < 0 || height < 0) throw std::invalid_argument("Negative output size");
        if (!width || !height) {
            if (fx <= 0 || fy <= 0) throw std::invalid_argument("Resize requires a positive size or positive scales");
            const double w = source.cols * fx, h = source.rows * fy;
            if (!std::isfinite(w) || !std::isfinite(h) || w > INT_MAX || h > INT_MAX)
                throw std::length_error("Resize dimensions overflow");
            width = cvRound(w); height = cvRound(h);
        }
        image_bytes(width, height, source.type());
        cv::resize(source, output(dst), kernel(width, height), 0, 0, interpolation);
    });
}
int cv_cvtColor(const cv_Mat* src, cv_Mat* dst, int code) {
    return status([&] { cv::cvtColor(mat(src), output(dst), code); bounded(mat(dst)); });
}
double cv_threshold(const cv_Mat* src, cv_Mat* dst, double threshold, double maxval, int type) {
    return guard<double>(NAN, [&] { check_finite(threshold); check_finite(maxval); return cv::threshold(mat(src), output(dst), threshold, maxval, type); });
}
cv_Mat* cv_imread(const char* filename, int flags) {
    return guard<cv_Mat*>(nullptr, [&] {
        auto image = cv::imread(text(filename), flags);
        if (image.empty()) throw std::runtime_error("OpenCV could not decode image");
        bounded(image); return new cv_Mat{std::move(image)};
    });
}
int cv_imwrite(const char* filename, const cv_Mat* image) {
    return guard<int>(-1, [&] {
        if (!cv::imwrite(text(filename), mat(image))) throw std::runtime_error("OpenCV could not write image");
        return 1;
    });
}
cv_VideoCapture* cv_VideoCapture_new(void) {
    return guard<cv_VideoCapture*>(nullptr, [] { return new cv_VideoCapture{}; });
}
cv_VideoCapture* cv_VideoCapture_new_index(int index) {
    return guard<cv_VideoCapture*>(nullptr, [&] {
        cv::VideoCapture value(index);
        if (!value.isOpened()) throw std::runtime_error("OpenCV could not open camera");
        return new cv_VideoCapture{std::move(value)};
    });
}
cv_VideoCapture* cv_VideoCapture_new_filename(const char* filename) {
    return guard<cv_VideoCapture*>(nullptr, [&] {
        cv::VideoCapture value(text(filename));
        if (!value.isOpened()) throw std::runtime_error("OpenCV could not open video");
        return new cv_VideoCapture{std::move(value)};
    });
}
void cv_VideoCapture_delete(cv_VideoCapture* value) { status([&] { delete value; }); }
int cv_VideoCapture_isOpened(const cv_VideoCapture* value) {
    return guard<int>(-1, [&] { return int(require(value).value.isOpened()); });
}
int cv_VideoCapture_read(cv_VideoCapture* value, cv_Mat* image) {
    return guard<int>(-1, [&] {
        bool read = require(value).value.read(output(image)); bounded(mat(image)); return int(read);
    });
}
void cv_VideoCapture_release(cv_VideoCapture* value) { status([&] { require(value).value.release(); }); }
int cv_namedWindow(const char* name, int flags) { return status([&] { gui(); cv::namedWindow(text(name), flags); }); }
int cv_destroyWindow(const char* name) { return status([&] { gui(); cv::destroyWindow(text(name)); }); }
int cv_destroyAllWindows(void) { return status([] { gui(); cv::destroyAllWindows(); }); }
int cv_imshow(const char* name, const cv_Mat* image) { return status([&] { gui(); cv::imshow(text(name), mat(image)); }); }
int cv_waitKey(int delay) { return guard<int>(-1, [&] { gui(); return cv::waitKey(delay); }); }
cv_SIFT* cv_SIFT_create(void) { return guard<cv_SIFT*>(nullptr, [] { return new cv_SIFT{cv::SIFT::create()}; }); }
void cv_SIFT_delete(cv_SIFT* detector) { status([&] { delete detector; }); }
int cv_SIFT_detectAndCompute(cv_SIFT* detector, const cv_Mat* image, const cv_Mat* mask, cv_KeyPoint_vector* points, cv_Mat* descriptors) {
    return status([&] { require(detector).value->detectAndCompute(mat(image), mask ? mat(mask) : cv::Mat{}, require(points).value, output(descriptors)); });
}
cv_CascadeClassifier* cv_CascadeClassifier_new(void) { return guard<cv_CascadeClassifier*>(nullptr, [] { return new cv_CascadeClassifier{}; }); }
cv_CascadeClassifier* cv_CascadeClassifier_new_filename(const char* filename) {
    return guard<cv_CascadeClassifier*>(nullptr, [&] {
        // Keep the object owned while load can throw or fail.
        std::unique_ptr<cv_CascadeClassifier> value(new cv_CascadeClassifier{});
        if (!value->value.load(text(filename))) throw std::runtime_error("OpenCV could not load cascade model");
        return value.release();
    });
}
void cv_CascadeClassifier_delete(cv_CascadeClassifier* value) { status([&] { delete value; }); }
int cv_CascadeClassifier_detectMultiScale(cv_CascadeClassifier* value, const cv_Mat* image, cv_Rect_vector* objects) {
    return status([&] { require(value).value.detectMultiScale(mat(image), require(objects).value); });
}
double cv_calibrateCamera(const cv_Point3f_vector_vector* objects, const cv_Point2f_vector_vector* images, cv_Size size, cv_Mat* camera, cv_Mat* distortion) {
    return guard<double>(NAN, [&] {
        std::vector<cv::Mat> rotations, translations;
        return cv::calibrateCamera(require(objects).value, require(images).value, kernel(size.width, size.height), output(camera), output(distortion), rotations, translations);
    });
}
int cv_solvePnP(const cv_Point3f_vector* objects, const cv_Point2f_vector* images, const cv_Mat* camera, const cv_Mat* distortion, cv_Mat* rotation, cv_Mat* translation) {
    return guard<int>(-1, [&] {
        return int(cv::solvePnP(require(objects).value, require(images).value, mat(camera), distortion ? mat(distortion) : cv::Mat{}, output(rotation), output(translation)));
    });
}
cv_ml_SVM* cv_ml_SVM_create(void) { return guard<cv_ml_SVM*>(nullptr, [] { return new cv_ml_SVM{cv::ml::SVM::create()}; }); }
void cv_ml_SVM_delete(cv_ml_SVM* value) { status([&] { delete value; }); }
int cv_ml_SVM_set_type(cv_ml_SVM* value, int type) { return status([&] { require(value).value->setType(type); }); }
int cv_ml_SVM_set_kernel(cv_ml_SVM* value, int kernel) { return status([&] { require(value).value->setKernel(kernel); }); }
int cv_ml_SVM_set_c(cv_ml_SVM* value, double c) { return status([&] { check_finite(c); require(value).value->setC(c); }); }
int cv_ml_SVM_train(cv_ml_SVM* value, const cv_Mat* samples, int layout, const cv_Mat* responses) {
    return guard<int>(-1, [&] { return int(require(value).value->train(mat(samples), layout, mat(responses))); });
}
double cv_ml_SVM_predict(cv_ml_SVM* value, const cv_Mat* samples, cv_Mat* results) {
    return guard<double>(NAN, [&] { return double(require(value).value->predict(mat(samples), output(results))); });
}
}
