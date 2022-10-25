package com.waynestalk.serviceexample

import android.app.Service
import android.content.Intent
import android.os.*
import android.util.Log

class DownloadService : Service() {
    companion object {
        const val ARGUMENT_RECEIVER = "receiver"
        const val RESULT_CODE_UPDATE_PROGRESS = 6
        const val RESULT_PROGRESS = "progress"

        private const val TAG = "DownloadService"
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

            stopSelf(msg.arg1)
        }
    }

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

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand=$startId => ${Thread.currentThread()}")

        receiver = intent?.getParcelableExtra(ARGUMENT_RECEIVER)

        serviceHandler?.obtainMessage()?.let { message ->
            message.arg1 = startId
            serviceHandler?.sendMessage(message)
        }

        return START_NOT_STICKY
    }

    override fun onBind(p0: Intent?): IBinder? {
        TODO("Not yet implemented")
    }
}