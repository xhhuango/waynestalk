package com.waynestalk.androidyolov8torchscriptexample

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
import org.pytorch.IValue
import org.pytorch.LitePyTorchAndroid
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils

class AppViewModel : ViewModel() {
    companion object {
        private const val IMAGE_WIDTH = 640
        private const val IMAGE_HEIGHT = 640
    }

    private val _uiState = MutableStateFlow(AppUiState())
    val uiState: StateFlow<AppUiState> = _uiState.asStateFlow()

    private var module: Module? = null

    fun load(assets: AssetManager) {
        viewModelScope.launch(Dispatchers.IO) {
            module = LitePyTorchAndroid.loadModuleFromAsset(assets, "baby_penguin.torchscript")

            assets.open("image.jpg").use {
                _uiState.value = AppUiState(Bitmap.createBitmap(BitmapFactory.decodeStream(it)))
            }
        }
    }

    fun infer(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.Default) {
            val image = Bitmap.createScaledBitmap(bitmap, IMAGE_WIDTH, IMAGE_HEIGHT, false)
            val input = TensorImageUtils.bitmapToFloat32Tensor(
                image,
                floatArrayOf(0.0f, 0.0f, 0.0f),
                floatArrayOf(1.0f, 1.0f, 1.0f),
            )
            val output = module?.forward(IValue.from(input))?.toTensor()
                ?: throw Exception("Module is not loaded")

            val boxes = PostProcessor.nms(tensor = output)
            _uiState.value = _uiState.value.copy(boxes = boxes)
        }
    }

    fun scale(canvasSize: Size, boxesSize: Size): Size {
        val scaleX = canvasSize.width / IMAGE_WIDTH
        val scaleY = canvasSize.height / IMAGE_HEIGHT
        return Size(boxesSize.width * scaleX, boxesSize.height * scaleY)
    }
}