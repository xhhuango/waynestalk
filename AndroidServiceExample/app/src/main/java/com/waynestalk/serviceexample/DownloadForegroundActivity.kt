package com.waynestalk.serviceexample

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.ResultReceiver
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.waynestalk.serviceexample.databinding.DownloadForegroundActivityBinding

class DownloadForegroundActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "ForegroundActivity"
    }

    private var _binding: DownloadForegroundActivityBinding? = null
    private val binding: DownloadForegroundActivityBinding
        get() = _binding!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        _binding = DownloadForegroundActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.downloadWithForegroundService.setOnClickListener {
            val intent = Intent(this, DownloadForegroundService::class.java)
            intent.putExtra(
                DownloadForegroundService.ARGUMENT_RECEIVER,
                UpdateReceiver(Handler(mainLooper))
            )
            startService(intent)
        }
    }

    inner class UpdateReceiver constructor(handler: Handler) : ResultReceiver(handler) {
        override fun onReceiveResult(resultCode: Int, resultData: Bundle) {
            super.onReceiveResult(resultCode, resultData)
            if (resultCode == DownloadForegroundService.RESULT_CODE_UPDATE_PROGRESS) {
                val progress = resultData.getInt(DownloadForegroundService.RESULT_PROGRESS)
                Log.d(TAG, "progress=" + progress + " => " + Thread.currentThread())
                binding.progress.text = "$progress %"
            }
        }
    }
}