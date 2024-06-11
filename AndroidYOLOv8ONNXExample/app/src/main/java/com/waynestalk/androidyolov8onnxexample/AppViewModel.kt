package com.waynestalk.androidyolov8onnxexample

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.ui.geometry.Size
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.BufferedInputStream
import java.nio.FloatBuffer
import java.util.Collections

class AppViewModel : ViewModel() {
    companion object {
        private const val IMAGE_WIDTH = 640
        private const val IMAGE_HEIGHT = 640

        private const val BATCH_SIZE = 1
        private const val PIXEL_SIZE = 3
    }

    private val _uiState = MutableStateFlow(AppUiState())
    val uiState: StateFlow<AppUiState> = _uiState.asStateFlow()

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    fun load(assets: AssetManager) {
        viewModelScope.launch(Dispatchers.IO) {
            ortEnv.use {
                assets.open( "baby_penguin.onnx").use {
                    BufferedInputStream(it).use { bis ->
                        session = ortEnv.createSession(bis.readBytes())
                    }
                }
            }

            assets.open("image.jpg").use {
                _uiState.value = AppUiState(Bitmap.createBitmap(BitmapFactory.decodeStream(it)))
            }
        }
    }

    fun infer(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.Default) {
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_WIDTH, IMAGE_HEIGHT, false)
            val input = createTensor(scaledBitmap)
            val model = session ?: throw Exception("Model is not set")
            val inputName = model.inputNames.iterator().next()
            val output = model.run(Collections.singletonMap(inputName, input))
            val boxes = PostProcessor.nms(output)
            _uiState.value = _uiState.value.copy(boxes = boxes)
        }
    }

    private fun createTensor(bitmap: Bitmap): OnnxTensor {
        ortEnv.use {
            return OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(bitmapToFlatArray(bitmap)),
                longArrayOf(
                    BATCH_SIZE.toLong(),
                    PIXEL_SIZE.toLong(),
                    bitmap.width.toLong(),
                    bitmap.height.toLong(),
                )
            )
        }
    }

    private fun bitmapToFlatArray(bitmap: Bitmap): FloatArray {
        val input = FloatArray(BATCH_SIZE * PIXEL_SIZE * bitmap.width * bitmap.height)

        val pixels = bitmap.width * bitmap.height
        val bitmapArray = IntArray(pixels)
        bitmap.getPixels(bitmapArray, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (i in 0..<bitmap.width) {
            for (j in 0..<bitmap.height) {
                val idx = bitmap.height * i + j
                val pixelValue = bitmapArray[idx]
                input[idx] = (pixelValue shr 16 and 0xFF) / 255f
                input[idx + pixels] = (pixelValue shr 8 and 0xFF) / 255f
                input[idx + pixels * 2] = (pixelValue and 0xFF) / 255f
            }
        }

        return input
    }

    fun scale(canvasSize: Size, boxesSize: Size): Size {
        val scaleX = canvasSize.width / IMAGE_WIDTH
        val scaleY = canvasSize.height / IMAGE_HEIGHT
        return Size(boxesSize.width * scaleX, boxesSize.height * scaleY)
    }
}