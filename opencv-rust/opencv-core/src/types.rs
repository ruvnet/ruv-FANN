//! Common type definitions for OpenCV

#![allow(non_camel_case_types)]

use crate::{MatType, Point, Point2f, Point2d, Point3f, Point3d, Size, Size2f, Size2d, Rect, Rect2f, Rect2d};

/// Common type aliases
pub type CV_8U = u8;
pub type CV_8S = i8;
pub type CV_16U = u16;
pub type CV_16S = i16;
pub type CV_32S = i32;
pub type CV_32F = f32;
pub type CV_64F = f64;

/// Color conversion codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ColorConversionCodes {
    COLOR_BGR2BGRA = 0,
    COLOR_BGRA2BGR = 1,
    COLOR_BGR2RGBA = 2,
    COLOR_RGBA2BGR = 3,
    COLOR_BGR2RGB = 4,
    COLOR_BGRA2RGBA = 5,
    COLOR_BGR2GRAY = 6,
    COLOR_RGB2GRAY = 7,
    COLOR_GRAY2BGR = 8,
    COLOR_GRAY2BGRA = 9,
    COLOR_BGRA2GRAY = 10,
    COLOR_RGBA2GRAY = 11,
}

/// Interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum InterpolationFlags {
    INTER_NEAREST = 0,
    INTER_LINEAR = 1,
    INTER_CUBIC = 2,
    INTER_AREA = 3,
    INTER_LANCZOS4 = 4,
    INTER_LINEAR_EXACT = 5,
    INTER_NEAREST_EXACT = 6,
    INTER_MAX = 7,
}

/// Threshold types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ThresholdTypes {
    THRESH_BINARY = 0,
    THRESH_BINARY_INV = 1,
    THRESH_TRUNC = 2,
    THRESH_TOZERO = 3,
    THRESH_TOZERO_INV = 4,
    THRESH_MASK = 7,
    THRESH_OTSU = 8,
    THRESH_TRIANGLE = 16,
}

/// Border types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BorderTypes {
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
    BORDER_REFLECT = 2,
    BORDER_WRAP = 3,
    BORDER_REFLECT_101 = 4,
    BORDER_TRANSPARENT = 5,
    BORDER_ISOLATED = 16,
}

/// Morphological operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MorphTypes {
    MORPH_ERODE = 0,
    MORPH_DILATE = 1,
    MORPH_OPEN = 2,
    MORPH_CLOSE = 3,
    MORPH_GRADIENT = 4,
    MORPH_TOPHAT = 5,
    MORPH_BLACKHAT = 6,
    MORPH_HITMISS = 7,
}

/// Image read modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ImreadModes {
    IMREAD_UNCHANGED = -1,
    IMREAD_GRAYSCALE = 0,
    IMREAD_COLOR = 1,
    IMREAD_ANYDEPTH = 2,
    IMREAD_ANYCOLOR = 4,
    IMREAD_LOAD_GDAL = 8,
    IMREAD_REDUCED_GRAYSCALE_2 = 16,
    IMREAD_REDUCED_COLOR_2 = 17,
    IMREAD_REDUCED_GRAYSCALE_4 = 32,
    IMREAD_REDUCED_COLOR_4 = 33,
    IMREAD_REDUCED_GRAYSCALE_8 = 64,
    IMREAD_REDUCED_COLOR_8 = 65,
    IMREAD_IGNORE_ORIENTATION = 128,
}

/// Video capture properties
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum VideoCaptureProperties {
    CAP_PROP_POS_MSEC = 0,
    CAP_PROP_POS_FRAMES = 1,
    CAP_PROP_POS_AVI_RATIO = 2,
    CAP_PROP_FRAME_WIDTH = 3,
    CAP_PROP_FRAME_HEIGHT = 4,
    CAP_PROP_FPS = 5,
    CAP_PROP_FOURCC = 6,
    CAP_PROP_FRAME_COUNT = 7,
    CAP_PROP_FORMAT = 8,
    CAP_PROP_MODE = 9,
    CAP_PROP_BRIGHTNESS = 10,
    CAP_PROP_CONTRAST = 11,
    CAP_PROP_SATURATION = 12,
    CAP_PROP_HUE = 13,
    CAP_PROP_GAIN = 14,
    CAP_PROP_EXPOSURE = 15,
    CAP_PROP_CONVERT_RGB = 16,
    CAP_PROP_WHITE_BALANCE_BLUE_U = 17,
    CAP_PROP_RECTIFICATION = 18,
    CAP_PROP_MONOCHROME = 19,
}

/// Video capture APIs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum VideoCaptureAPIs {
    CAP_ANY = 0,
    CAP_VFW = 200,
    CAP_V4L2 = 201,
    CAP_FIREWIRE = 300,
    CAP_QT = 500,
    CAP_UNICAP = 600,
    CAP_DSHOW = 700,
    CAP_PVAPI = 800,
    CAP_OPENNI = 900,
    CAP_OPENNI_ASUS = 910,
    CAP_ANDROID = 1000,
    CAP_XIAPI = 1100,
    CAP_AVFOUNDATION = 1200,
    CAP_GIGANETIX = 1300,
    CAP_MSMF = 1400,
    CAP_WINRT = 1410,
    CAP_INTELPERC = 1500,
    CAP_OPENNI2 = 1600,
    CAP_OPENNI2_ASUS = 1610,
    CAP_OPENNI2_ASTRA = 1620,
    CAP_GPHOTO2 = 1700,
    CAP_GSTREAMER = 1800,
    CAP_FFMPEG = 1900,
    CAP_IMAGES = 2000,
    CAP_ARAVIS = 2100,
    CAP_OPENCV_MJPEG = 2200,
    CAP_INTEL_MFX = 2300,
    CAP_XINE = 2400,
}

/// Window flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WindowFlags {
    WINDOW_NORMAL = 0x00000000,
    WINDOW_AUTOSIZE = 0x00000001,
    WINDOW_OPENGL = 0x00001000,
    WINDOW_FULLSCREEN = 0x00000002,
    WINDOW_FREERATIO = 0x00000100,
    WINDOW_KEEPRATIO = 0x00000200,
    WINDOW_GUI_EXPANDED = 0x00000400,
    WINDOW_GUI_NORMAL = 0x00000010,
}

/// Mouse event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MouseEventTypes {
    EVENT_MOUSEMOVE = 0,
    EVENT_LBUTTONDOWN = 1,
    EVENT_RBUTTONDOWN = 2,
    EVENT_MBUTTONDOWN = 3,
    EVENT_LBUTTONUP = 4,
    EVENT_RBUTTONUP = 5,
    EVENT_MBUTTONUP = 6,
    EVENT_LBUTTONDBLCLK = 7,
    EVENT_RBUTTONDBLCLK = 8,
    EVENT_MBUTTONDBLCLK = 9,
    EVENT_MOUSEWHEEL = 10,
    EVENT_MOUSEHWHEEL = 11,
}

/// Mouse event flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MouseEventFlags {
    EVENT_FLAG_LBUTTON = 1,
    EVENT_FLAG_RBUTTON = 2,
    EVENT_FLAG_MBUTTON = 4,
    EVENT_FLAG_CTRLKEY = 8,
    EVENT_FLAG_SHIFTKEY = 16,
    EVENT_FLAG_ALTKEY = 32,
}