#include "bridge.hpp"
using namespace ruv_cv;
namespace {
cv::Point2f to_native(cv_Point2f p) { return {p.x, p.y}; }
cv::Point3f to_native(cv_Point3f p) { return {p.x, p.y, p.z}; }
cv::Rect to_native(cv_Rect r) { return {r.x, r.y, r.width, r.height}; }
cv::KeyPoint to_native(cv_KeyPoint p) { return {{p.pt.x,p.pt.y},p.size,p.angle,p.response,p.octave,p.class_id}; }
cv_Point2f to_c(cv::Point2f p) { return {p.x, p.y}; }
cv_Point3f to_c(cv::Point3f p) { return {p.x, p.y, p.z}; }
cv_Rect to_c(cv::Rect r) { return {r.x, r.y, r.width, r.height}; }
cv_KeyPoint to_c(cv::KeyPoint p) { return {{p.pt.x,p.pt.y},p.size,p.angle,p.response,p.octave,p.class_id}; }
}
extern "C" {
#define VECTOR_API(Name, Value) \
Name* Name##_new(void) { return guard<Name*>(nullptr, [] { return new Name{}; }); } \
void Name##_delete(Name* value) { status([&] { delete value; }); } \
size_t Name##_len(const Name* value) { return guard<size_t>(SIZE_MAX, [&] { return require(value).value.size(); }); } \
int Name##_clear(Name* value) { return status([&] { require(value).value.clear(); }); } \
int Name##_push(Name* value, Value item) { return status([&] { require(value).value.push_back(to_native(item)); }); } \
int Name##_get(const Name* value, size_t index, Value* out) { return status([&] { require(out) = to_c(require(value).value.at(index)); }); }
VECTOR_API(cv_KeyPoint_vector, cv_KeyPoint)
VECTOR_API(cv_Rect_vector, cv_Rect)
VECTOR_API(cv_Point2f_vector, cv_Point2f)
VECTOR_API(cv_Point3f_vector, cv_Point3f)
#undef VECTOR_API
#define NESTED_API(Name, Inner) \
Name* Name##_new(void) { return guard<Name*>(nullptr, [] { return new Name{}; }); } \
void Name##_delete(Name* value) { status([&] { delete value; }); } \
size_t Name##_len(const Name* value) { return guard<size_t>(SIZE_MAX, [&] { return require(value).value.size(); }); } \
int Name##_clear(Name* value) { return status([&] { require(value).value.clear(); }); } \
int Name##_push(Name* value, const Inner* item) { return status([&] { require(value).value.push_back(require(item).value); }); } \
Inner* Name##_get(const Name* value, size_t index) { return guard<Inner*>(nullptr, [&] { return new Inner{require(value).value.at(index)}; }); }
NESTED_API(cv_Point2f_vector_vector, cv_Point2f_vector)
NESTED_API(cv_Point3f_vector_vector, cv_Point3f_vector)
#undef NESTED_API
}
