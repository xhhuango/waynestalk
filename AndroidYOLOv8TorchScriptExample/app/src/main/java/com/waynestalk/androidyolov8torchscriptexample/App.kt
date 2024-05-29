package com.waynestalk.androidyolov8torchscriptexample

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.waynestalk.androidyolov8torchscriptexample.ui.theme.AppTheme

@Composable
fun App(viewModel: AppViewModel = viewModel()) {
    val context = LocalContext.current
    LaunchedEffect(Unit) {
        viewModel.load(context.assets)
    }

    val uiState by viewModel.uiState.collectAsState()

    Scaffold(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) { innerPadding ->
        Column(
            Modifier
                .padding(innerPadding)
                .fillMaxSize()
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .border(1.dp, MaterialTheme.colorScheme.primary)
            ) {
                val bitmap = uiState.bitmap
                if (bitmap != null) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.FillBounds,
                        modifier = Modifier.fillMaxSize()
                    )
                }

                Canvas(modifier = Modifier.fillMaxSize()) {
                    drawRect(Color.Transparent, size = size)
                    uiState.boxes.forEach {
                        it.boundingBox.also { rect ->
                            val offsetSize = viewModel.scale(size, Size(rect.left, rect.top))
                            val boxSize = viewModel.scale(
                                size,
                                Size(rect.right - rect.left, rect.bottom - rect.top)
                            )
                            drawRect(
                                Color.Green,
                                topLeft = Offset(offsetSize.width, offsetSize.height),
                                size = boxSize,
                                style = Stroke(width = 4f)
                            )
                        }
                    }
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp),
            ) {
                Button(
                    onClick = { viewModel.infer(uiState.bitmap ?: return@Button) },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(text = stringResource(id = R.string.infer))
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun AppPreview() {
    AppTheme {
        App()
    }
}
