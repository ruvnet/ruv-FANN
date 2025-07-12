// OpenCV C++ wrapper header for bindgen
// This file defines the C interface that bindgen will use to generate Rust bindings

#ifndef OPENCV_WRAPPER_H
#define OPENCV_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Basic types
typedef struct cv_Mat cv_Mat;
typedef struct cv_VideoCapture cv_VideoCapture;
typedef struct cv_Size { int width, height; } cv_Size;
typedef struct cv_Point2i { int x, y; } cv_Point2i;
typedef struct cv_Rect { int x, y, width, height; } cv_Rect;
typedef struct cv_Scalar { double val[4]; } cv_Scalar;

// Mat operations
cv_Mat* cv_Mat_new();
cv_Mat* cv_Mat_new_size_type_scalar(int rows, int cols, int type, cv_Scalar scalar);
void cv_Mat_delete(cv_Mat* mat);
cv_Mat* cv_Mat_clone(const cv_Mat* mat);
int cv_Mat_rows(const cv_Mat* mat);
int cv_Mat_cols(const cv_Mat* mat);
int cv_Mat_type(const cv_Mat* mat);
int cv_Mat_channels(const cv_Mat* mat);
int cv_Mat_empty(const cv_Mat* mat);
unsigned char* cv_Mat_data(const cv_Mat* mat);

// Image processing
int cv_blur(const cv_Mat* src, cv_Mat* dst, int ksize_width, int ksize_height);
int cv_GaussianBlur(const cv_Mat* src, cv_Mat* dst, int ksize_width, int ksize_height, double sigmaX, double sigmaY);
int cv_resize(const cv_Mat* src, cv_Mat* dst, int dsize_width, int dsize_height, double fx, double fy, int interpolation);

// Image I/O
cv_Mat* cv_imread(const char* filename, int flags);
int cv_imwrite(const char* filename, const cv_Mat* img);

// Video I/O
cv_VideoCapture* cv_VideoCapture_new();
cv_VideoCapture* cv_VideoCapture_new_index(int index);
void cv_VideoCapture_delete(cv_VideoCapture* cap);
int cv_VideoCapture_isOpened(const cv_VideoCapture* cap);
int cv_VideoCapture_read(cv_VideoCapture* cap, cv_Mat* image);

#ifdef __cplusplus
}
#endif

#endif // OPENCV_WRAPPER_H