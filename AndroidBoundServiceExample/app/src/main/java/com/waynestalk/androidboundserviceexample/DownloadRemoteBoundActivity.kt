package com.waynestalk.androidboundserviceexample

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.*
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.waynestalk.androidboundserviceexample.databinding.DownloadRemoteBoundActivityBinding

class DownloadRemoteBoundActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "RemoteBoundActivity"
    }

    inner class IncomingHandler(looper: Looper) : Handler(looper) {
        override fun handleMessage(msg: Message) {
            when (msg.what) {
                DownloadRemoteBoundService.RES_UPDATE_PROGRESS -> {
                    val progress = msg.arg1
                    Log.d(
                        TAG,
                        "progress=$progress => pid=${Process.myPid()} ${Thread.currentThread()}"
                    )
                    binding.progress.text = "$progress %"
                }
                DownloadRemoteBoundService.RES_UPDATE_COMPLETE -> {
                    unbindService(connection)
                }
                else -> super.handleMessage(msg)
            }
        }
    }

    private var _binding: DownloadRemoteBoundActivityBinding? = null
    private val binding: DownloadRemoteBoundActivityBinding
        get() = _binding!!

    private var service: Messenger? = null
    private var isBound = false

    private var messenger: Messenger? = null

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(className: ComponentName, binder: IBinder) {
            service = Messenger(binder)
            messenger = Messenger(IncomingHandler(mainLooper))
            isBound = true
        }

        override fun onServiceDisconnected(className: ComponentName) {
            service = null
            messenger = null
            isBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        _binding = DownloadRemoteBoundActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.downloadWithRemoteBoundService.setOnClickListener {
            if (isBound) {
                val msg = Message.obtain(null, DownloadRemoteBoundService.CMD_UPDATE, 0, 0)
                msg.replyTo = messenger
                try {
                    service?.send(msg)
                } catch (e: RemoteException) {
                    // The service is crashed.
                    e.printStackTrace()
                }
            } else {
                Log.d(TAG, "Service is not bound yet")
            }
        }
    }

    override fun onStart() {
        super.onStart()

        val intent = Intent(this, DownloadRemoteBoundService::class.java)
        bindService(intent, connection, Context.BIND_AUTO_CREATE)
    }

    override fun onStop() {
        super.onStop()

        if (isBound) {
            unbindService(connection)
            isBound = false
        }
    }
}