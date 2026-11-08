#ifndef RUV_OPENCV_WRAPPER_H
#define RUV_OPENCV_WRAPPER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Every handle is owned and opaque. Delete it once using its matching delete
 * function. NULL is accepted by delete; dangling or foreign pointers are invalid.
 * Calls catch C++ exceptions. Status: 0 success, -1 failure; boolean queries:
 * 0 false, 1 true, -1 failure. Pointer failures return NULL, floating failures
 * return NaN. cv_last_error() describes the last failure on the calling thread;
 * the message is valid until the next bridge operation on that thread.
 * Handles and borrowed data require external synchronization. Raw byte input
 * and output buffers must not overlap; inputs must be initialized and string
 * pointers must be NUL-terminated. */
typedef struct cv_Mat cv_Mat;
typedef struct cv_VideoCapture cv_VideoCapture;
typedef struct cv_SIFT cv_SIFT;
typedef struct cv_CascadeClassifier cv_CascadeClassifier;
typedef struct cv_ml_SVM cv_ml_SVM;
typedef struct cv_KeyPoint_vector cv_KeyPoint_vector;
typedef struct cv_Rect_vector cv_Rect_vector;
typedef struct cv_Point2f_vector cv_Point2f_vector;
typedef struct cv_Point3f_vector cv_Point3f_vector;
typedef struct cv_Point2f_vector_vector cv_Point2f_vector_vector;
typedef struct cv_Point3f_vector_vector cv_Point3f_vector_vector;
typedef struct cv_Size { int width, height; } cv_Size;
typedef struct cv_Point2i { int x, y; } cv_Point2i;
typedef struct cv_Point2f { float x, y; } cv_Point2f;
typedef struct cv_Point2d { double x, y; } cv_Point2d;
typedef struct cv_Point3f { float x, y, z; } cv_Point3f;
typedef struct cv_Rect { int x, y, width, height; } cv_Rect;
typedef struct cv_Scalar { double val[4]; } cv_Scalar;
typedef struct cv_KeyPoint {
    cv_Point2f pt;
    float size, angle, response;
    int octave, class_id;
} cv_KeyPoint;

const char* cv_last_error(void);
const char* cv_version(void);
cv_Mat* cv_Mat_new(void);
cv_Mat* cv_Mat_new_size_type_scalar(int rows, int cols, int type, cv_Scalar scalar);
/* Imports copy pixels. The legacy pointer API requires rows*cols*element_size
 * readable bytes. Prefer from_bytes, which validates the supplied byte count. */
cv_Mat* cv_Mat_from_data(const void* data, int rows, int cols, int type);
cv_Mat* cv_Mat_from_bytes(const unsigned char* data, size_t len, int rows, int cols, int type);
void cv_Mat_delete(cv_Mat* mat);
cv_Mat* cv_Mat_clone(const cv_Mat* mat);
int cv_Mat_copyTo(const cv_Mat* src, cv_Mat* dst);
int cv_Mat_convertTo(const cv_Mat* src, cv_Mat* dst, int type, double alpha, double beta);
int cv_Mat_rows(const cv_Mat* mat);
int cv_Mat_cols(const cv_Mat* mat);
int cv_Mat_type(const cv_Mat* mat);
int cv_Mat_channels(const cv_Mat* mat);
int cv_Mat_empty(const cv_Mat* mat);
int cv_Mat_is_continuous(const cv_Mat* mat);
size_t cv_Mat_step(const cv_Mat* mat);
size_t cv_Mat_total(const cv_Mat* mat);
size_t cv_Mat_element_size(const cv_Mat* mat);
/* Borrowed pointers expire on deletion/reallocation. ROI can be strided; use
 * row_data or copy_data rather than assuming rows*cols bytes are contiguous. */
unsigned char* cv_Mat_data(const cv_Mat* mat);
unsigned char* cv_Mat_row_data(const cv_Mat* mat, int row);
int cv_Mat_copy_data(const cv_Mat* mat, unsigned char* dst, size_t len);
/* Native ROI shares refcounted storage with its parent. It remains allocated
 * after parent deletion; writes alias the parent and require synchronization. */
cv_Mat* cv_Mat_roi(const cv_Mat* mat, int x, int y, int width, int height);

int cv_blur(const cv_Mat* src, cv_Mat* dst, int width, int height);
int cv_GaussianBlur(const cv_Mat* src, cv_Mat* dst, int width, int height, double sigma_x, double sigma_y);
int cv_Canny(const cv_Mat* src, cv_Mat* dst, double threshold1, double threshold2);
int cv_resize(const cv_Mat* src, cv_Mat* dst, int width, int height, double fx, double fy, int interpolation);
int cv_cvtColor(const cv_Mat* src, cv_Mat* dst, int code);
double cv_threshold(const cv_Mat* src, cv_Mat* dst, double threshold, double maxval, int type);
cv_Mat* cv_imread(const char* filename, int flags);
int cv_imwrite(const char* filename, const cv_Mat* image);
cv_VideoCapture* cv_VideoCapture_new(void);
cv_VideoCapture* cv_VideoCapture_new_index(int index);
cv_VideoCapture* cv_VideoCapture_new_filename(const char* filename);
void cv_VideoCapture_delete(cv_VideoCapture* capture);
int cv_VideoCapture_isOpened(const cv_VideoCapture* capture);
int cv_VideoCapture_read(cv_VideoCapture* capture, cv_Mat* image);
void cv_VideoCapture_release(cv_VideoCapture* capture);
/* GUI requires a running display and OpenCV's main-thread UI discipline. */
int cv_namedWindow(const char* name, int flags);
int cv_destroyWindow(const char* name);
int cv_destroyAllWindows(void);
int cv_imshow(const char* name, const cv_Mat* image);
int cv_waitKey(int delay);
cv_SIFT* cv_SIFT_create(void);
void cv_SIFT_delete(cv_SIFT* detector);
int cv_SIFT_detectAndCompute(cv_SIFT* detector, const cv_Mat* image, const cv_Mat* mask, cv_KeyPoint_vector* points, cv_Mat* descriptors);
cv_CascadeClassifier* cv_CascadeClassifier_new(void);
cv_CascadeClassifier* cv_CascadeClassifier_new_filename(const char* filename);
void cv_CascadeClassifier_delete(cv_CascadeClassifier* classifier);
int cv_CascadeClassifier_detectMultiScale(cv_CascadeClassifier* classifier, const cv_Mat* image, cv_Rect_vector* objects);
double cv_calibrateCamera(const cv_Point3f_vector_vector* object_points, const cv_Point2f_vector_vector* image_points, cv_Size size, cv_Mat* camera, cv_Mat* distortion);
int cv_solvePnP(const cv_Point3f_vector* object_points, const cv_Point2f_vector* image_points, const cv_Mat* camera, const cv_Mat* distortion, cv_Mat* rotation, cv_Mat* translation);
cv_ml_SVM* cv_ml_SVM_create(void);
void cv_ml_SVM_delete(cv_ml_SVM* svm);
int cv_ml_SVM_set_type(cv_ml_SVM* svm, int type);
int cv_ml_SVM_set_kernel(cv_ml_SVM* svm, int kernel);
int cv_ml_SVM_set_c(cv_ml_SVM* svm, double c);
int cv_ml_SVM_train(cv_ml_SVM* svm, const cv_Mat* samples, int layout, const cv_Mat* responses);
double cv_ml_SVM_predict(cv_ml_SVM* svm, const cv_Mat* samples, cv_Mat* results);

/* Vector pushes copy values, including nested vectors. get validates bounds.
 * Length queries return SIZE_MAX on error (with cv_last_error set). */
#define CV_DECLARE_VECTOR(Name, Value) \
    Name* Name##_new(void); \
    void Name##_delete(Name* vector); \
    size_t Name##_len(const Name* vector); \
    int Name##_clear(Name* vector); \
    int Name##_push(Name* vector, Value value); \
    int Name##_get(const Name* vector, size_t index, Value* out);
CV_DECLARE_VECTOR(cv_KeyPoint_vector, cv_KeyPoint)
CV_DECLARE_VECTOR(cv_Rect_vector, cv_Rect)
CV_DECLARE_VECTOR(cv_Point2f_vector, cv_Point2f)
CV_DECLARE_VECTOR(cv_Point3f_vector, cv_Point3f)
#undef CV_DECLARE_VECTOR
#define CV_DECLARE_NESTED(Name, Inner) \
    Name* Name##_new(void); \
    void Name##_delete(Name* vector); \
    size_t Name##_len(const Name* vector); \
    int Name##_clear(Name* vector); \
    int Name##_push(Name* vector, const Inner* value); \
    Inner* Name##_get(const Name* vector, size_t index);
CV_DECLARE_NESTED(cv_Point2f_vector_vector, cv_Point2f_vector)
CV_DECLARE_NESTED(cv_Point3f_vector_vector, cv_Point3f_vector)
#undef CV_DECLARE_NESTED

#ifdef RUV_OPENCV_CUDA
typedef struct cv_cuda_GpuMat cv_cuda_GpuMat;
cv_cuda_GpuMat* cv_cuda_GpuMat_new(void);
void cv_cuda_GpuMat_delete(cv_cuda_GpuMat* mat);
int cv_cuda_GpuMat_upload(cv_cuda_GpuMat* gpu, const cv_Mat* host);
int cv_cuda_GpuMat_download(const cv_cuda_GpuMat* gpu, cv_Mat* host);
int cv_cuda_blur(const cv_cuda_GpuMat* src, cv_cuda_GpuMat* dst, int width, int height);
int cv_cuda_GaussianBlur(const cv_cuda_GpuMat* src, cv_cuda_GpuMat* dst, int width, int height, double sigma_x, double sigma_y);
#endif
#ifdef __cplusplus
}
#endif
#endif
