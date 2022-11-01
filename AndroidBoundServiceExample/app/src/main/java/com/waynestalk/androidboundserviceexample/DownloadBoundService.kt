package com.waynestalk.androidboundserviceexample

import android.app.Service
import android.content.Intent
import android.os.*
import android.util.Log

class DownloadBoundService : Service() {
    companion object {
        const val RESULT_CODE_UPDATE_PROGRESS = 7
        const val RESULT_PROGRESS = "progress"

        private const val TAG = "DownloadBoundService"
    }

    inner class ServiceHandler(looper: Looper) : Handler(looper) {
        override fun handleMessage(msg: Message) {
            Log.d(TAG, "handleMessage: msg.arg1=${msg.arg1} => ${Thread.currentThread()}")

            for (i in 1..10) {
                try {
                    Thread.sleep(1000)

                    val resultData = Bundle().apply {
                        putInt(RESULT_PROGRESS, i * 10)
                    }
                    receiver?.send(RESULT_CODE_UPDATE_PROGRESS, resultData)
                } catch (e: InterruptedException) {
                    e.printStackTrace()
                }
            }
        }
    }

    inner class DownloadBinder : Binder() {
        val service: DownloadBoundService
            get() = this@DownloadBoundService
    }

    private val binder = DownloadBinder()

    private var looper: Looper? = null
    private var serviceHandler: ServiceHandler? = null

    private var receiver: ResultReceiver? = null

    override fun onCreate() {
        Log.d(TAG, "onCreate => ${Thread.currentThread()}")

        val handlerThread = HandlerThread("DownloadService", Process.THREAD_PRIORITY_BACKGROUND)
        handlerThread.start()

        looper = handlerThread.looper
        serviceHandler = ServiceHandler(handlerThread.looper)
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "onDestroy => ${Thread.currentThread()}")
    }

    override fun onBind(intent: Intent): IBinder {
        Log.d(TAG, "onBind => ${Thread.currentThread()}")
        return binder
    }

    fun startDownload(receiver: ResultReceiver) {
        this.receiver = receiver

        serviceHandler?.obtainMessage()?.let { message ->
            message.arg1 = 0
            serviceHandler?.sendMessage(message)
        }
    }
}