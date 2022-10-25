package com.waynestalk.serviceexample

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.ResultReceiver
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.waynestalk.serviceexample.databinding.DownloadActivityBinding

class DownloadActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "DownloadActivity"
    }

    private var _binding: DownloadActivityBinding? = null
    private val binding: DownloadActivityBinding
        get() = _binding!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        _binding = DownloadActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.downloadWithStartCommand.setOnClickListener {
            val intent = Intent(this, DownloadService::class.java)
            intent.putExtra(DownloadService.ARGUMENT_RECEIVER, UpdateReceiver(Handler(mainLooper)))
            startService(intent)
        }
    }

    inner class UpdateReceiver constructor(handler: Handler) : ResultReceiver(handler) {
        override fun onReceiveResult(resultCode: Int, resultData: Bundle) {
            super.onReceiveResult(resultCode, resultData)
            if (resultCode == DownloadService.RESULT_CODE_UPDATE_PROGRESS) {
                val progress = resultData.getInt(DownloadService.RESULT_PROGRESS)
                Log.d(TAG, "progress=" + progress + " => " + Thread.currentThread())
                binding.progress.text = "$progress %"
            }
        }
    }
}