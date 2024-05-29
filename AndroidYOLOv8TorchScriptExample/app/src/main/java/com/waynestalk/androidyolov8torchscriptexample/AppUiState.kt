package com.waynestalk.androidyolov8torchscriptexample

import android.graphics.Bitmap

data class AppUiState(
    val bitmap: Bitmap? = null,
    val boxes: List<BoundingBox> = emptyList(),
)