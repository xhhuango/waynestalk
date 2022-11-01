package com.waynestalk.androidboundserviceexample

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.ResultReceiver
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.waynestalk.androidboundserviceexample.databinding.DownloadBoundActivityBinding

class DownloadBoundActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "DownloadBoundActivity"
    }

    private var _binding: DownloadBoundActivityBinding? = null
    private val binding: DownloadBoundActivityBinding
        get() = _binding!!

    private var service: DownloadBoundService? = null
    private var isBound = false

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(className: ComponentName, binder: IBinder) {
            Log.d(TAG, "onServiceConnected: className=${className} => ${Thread.currentThread()}")
            val serviceBinder = binder as DownloadBoundService.DownloadBinder
            service = serviceBinder.service
            isBound = true
        }

        override fun onServiceDisconnected(className: ComponentName) {
            service = null
            isBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        _binding = DownloadBoundActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.downloadWithBoundService.setOnClickListener {
            if (isBound) {
                service?.startDownload(UpdateReceiver(Handler(mainLooper)))
            } else {
                Log.d(TAG, "Service is not bound yet")
            }
        }
    }

    override fun onStart() {
        super.onStart()

        val intent = Intent(this, DownloadBoundService::class.java)
        bindService(intent, connection, Context.BIND_AUTO_CREATE)
    }

    override fun onStop() {
        super.onStop()

        if (isBound) {
            unbindService(connection)
            isBound = false
        }
    }

    inner class UpdateReceiver constructor(handler: Handler) : ResultReceiver(handler) {
        override fun onReceiveResult(resultCode: Int, resultData: Bundle) {
            super.onReceiveResult(resultCode, resultData)
            if (resultCode == DownloadBoundService.RESULT_CODE_UPDATE_PROGRESS) {
                val progress = resultData.getInt(DownloadBoundService.RESULT_PROGRESS)
                Log.d(TAG, "progress=" + progress + " => " + Thread.currentThread())
                binding.progress.text = "$progress %"
            }
        }
    }
}