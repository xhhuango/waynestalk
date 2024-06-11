package com.waynestalk.androidyolov8onnxexample

import android.graphics.Bitmap

data class AppUiState(
    val bitmap: Bitmap? = null,
    val boxes: List<BoundingBox> = emptyList(),
)