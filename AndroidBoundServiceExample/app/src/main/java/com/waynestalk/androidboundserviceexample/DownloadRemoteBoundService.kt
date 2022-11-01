package com.waynestalk.androidboundserviceexample

import android.app.Service
import android.content.Intent
import android.os.*
import android.util.Log

class DownloadRemoteBoundService : Service() {
    companion object {
        const val CMD_UPDATE = 1

        const val RES_UPDATE_PROGRESS = 2
        const val RES_UPDATE_COMPLETE = 3

        private const val TAG = "RemoteBoundService"
    }

    inner class IncomingHandler(looper: Looper) : Handler(looper) {
        override fun handleMessage(msg: Message) {
            Log.d(
                TAG,
                "handleMessage: msg.what=${msg.what} => pid=${Process.myPid()}, ${Thread.currentThread()}"
            )

            when (msg.what) {
                CMD_UPDATE -> {
                    for (i in 1..10) {
                        try {
                            Thread.sleep(1000)

                            if (!send(msg.replyTo, RES_UPDATE_PROGRESS, i * 10)) return
                        } catch (e: InterruptedException) {
                            e.printStackTrace()
                            send(msg.replyTo, RES_UPDATE_COMPLETE, 0)
                            return
                        }
                    }

                    send(msg.replyTo, RES_UPDATE_COMPLETE, 1)
                }
                else -> super.handleMessage(msg)
            }
        }

        private fun send(replyTo: Messenger, command: Int, result: Int): Boolean {
            return try {
                val resultMsg = Message.obtain(null, command, result, 0)
                replyTo.send(resultMsg)
                true
            } catch (e: RemoteException) {
                // The client is dead.
                e.printStackTrace()
                false
            }
        }
    }

    private var looper: Looper? = null
    private lateinit var messenger: Messenger

    override fun onCreate() {
        Log.d(TAG, "onCreate => pid=${Process.myPid()}, ${Thread.currentThread()}")

        val handlerThread =
            HandlerThread("DownloadRemoteService", Process.THREAD_PRIORITY_BACKGROUND)
        handlerThread.start()

        looper = handlerThread.looper
        messenger = Messenger(IncomingHandler(handlerThread.looper))
    }

    override fun onDestroy() {
        Log.d(TAG, "onDestroy => pid=${Process.myPid()}, ${Thread.currentThread()}")
    }

    override fun onBind(intent: Intent): IBinder? {
        return messenger.binder
    }
}