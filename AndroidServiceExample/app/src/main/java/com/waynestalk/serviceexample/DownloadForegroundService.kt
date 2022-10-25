package com.waynestalk.serviceexample

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.*
import android.util.Log
import androidx.core.app.NotificationCompat

class DownloadForegroundService : Service() {
    companion object {
        const val ARGUMENT_RECEIVER = "receiver"
        const val RESULT_CODE_UPDATE_PROGRESS = 8
        const val RESULT_PROGRESS = "progress"

        private const val CHANNEL_ID = "DownloadForegroundServiceChannel"

        private const val TAG = "ForegroundService"
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

        createNotificationChannel()
        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, 0)
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Download Foreground Service")
            .setContentText("Downloading ...")
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
        startForeground(startId, notification)

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

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Download Foreground Service",
                NotificationManager.IMPORTANCE_DEFAULT
            )
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }
}