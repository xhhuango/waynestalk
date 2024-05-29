package com.waynestalk.androidyolov8torchscriptexample

import android.graphics.RectF
import org.pytorch.Tensor
import java.util.Arrays
import kotlin.math.max
import kotlin.math.min

data class BoundingBox(val boundingBox: RectF, var score: Float, var clazz: Int)

object PostProcessor {
    private const val CONFIDENCE_THRESHOLD = 0.25f
    private const val IOU_THRESHOLD = 0.45f
    private const val MAX_NMS = 30000

    fun nms(
        tensor: Tensor,
        confidenceThreshold: Float = CONFIDENCE_THRESHOLD,
        iouThreshold: Float = IOU_THRESHOLD,
        maxNms: Int = MAX_NMS,
        scaleX: Float = 1f,
        scaleY: Float = 1f,
    ): List<BoundingBox> {
        val array = tensor.dataAsFloatArray
        val rows = tensor.shape()[1].toInt()
        val cols = tensor.shape()[2].toInt()
        val outputs = Array(rows) { row -> array.sliceArray((row * cols) until ((row + 1) * cols)) }

        val results = mutableListOf<BoundingBox>()

        for (col in 0 until outputs[0].size) {
            var score = 0f
            var cls = 0
            for (row in 4 until outputs.size) {
                if (outputs[row][col] > score) {
                    score = outputs[row][col]
                    cls = row
                }
            }
            cls -= 4

            if (score > confidenceThreshold) {
                val x = outputs[0][col]
                val y = outputs[1][col]
                val w = outputs[2][col]
                val h = outputs[3][col]

                val left = x - w / 2
                val top = y - h / 2
                val right = x + w / 2
                val bottom = y + h / 2

                val rect = RectF(scaleX * left, top * scaleY, scaleX * right, scaleY * bottom)
                val result = BoundingBox(rect, score, cls)
                results.add(result)
            }
        }

        return nms(results, iouThreshold, maxNms)
    }

    private fun nms(boxes: List<BoundingBox>, iouThreshold: Float, limit: Int): List<BoundingBox> {
        val selected = mutableListOf<BoundingBox>()
        val sortedBoxes = boxes.sortedWith { o1, o2 -> o1.score.compareTo(o2.score) }
        val active = BooleanArray(sortedBoxes.size)
        Arrays.fill(active, true)
        var numActive = active.size

        var done = false
        var i = 0
        while (i < sortedBoxes.size && !done) {
            if (active[i]) {
                val boxA = sortedBoxes[i]
                selected.add(boxA)
                if (selected.size >= limit) break

                for (j in i + 1 until sortedBoxes.size) {
                    if (active[j]) {
                        val boxB = sortedBoxes[j]
                        if (calculateIou(boxA.boundingBox, boxB.boundingBox) > iouThreshold) {
                            active[j] = false
                            numActive -= 1
                            if (numActive <= 0) {
                                done = true
                                break
                            }
                        }
                    }
                }
            }
            i++
        }

        return selected
    }

    private fun calculateIou(a: RectF, b: RectF): Float {
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        if (areaA <= 0.0) return 0.0f

        val areaB = (b.right - b.left) * (b.bottom - b.top)
        if (areaB <= 0.0) return 0.0f

        val intersectionMinX = max(a.left, b.left)
        val intersectionMinY = max(a.top, b.top)
        val intersectionMaxX = min(a.right, b.right)
        val intersectionMaxY = min(a.bottom, b.bottom)
        val intersectionArea = max(
            intersectionMaxY - intersectionMinY, 0.0f
        ) * max(intersectionMaxX - intersectionMinX, 0.0f)

        return intersectionArea / (areaA + areaB - intersectionArea)
    }
}
