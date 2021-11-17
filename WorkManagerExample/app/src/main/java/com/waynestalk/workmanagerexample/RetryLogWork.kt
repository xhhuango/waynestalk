package com.waynestalk.workmanagerexample

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters

class RetryLogWork(appContext: Context, workerParameters: WorkerParameters) :
    CoroutineWorker(appContext, workerParameters) {
    companion object {
        const val MESSAGE = "message"
    }

    private val tag = javaClass.canonicalName

    override suspend fun doWork(): Result {
        val message = inputData.getString(MESSAGE) ?: return Result.failure()
        Log.d(tag, "$message, retry count: $runAttemptCount")
        return Result.retry()
    }
}