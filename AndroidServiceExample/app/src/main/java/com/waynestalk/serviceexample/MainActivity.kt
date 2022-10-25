package com.waynestalk.serviceexample

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.waynestalk.serviceexample.databinding.MainActivityBinding

class MainActivity : AppCompatActivity() {
    private var _binding: MainActivityBinding? = null
    private val binding: MainActivityBinding
        get() = _binding!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        _binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.downloadWithStartCommand.setOnClickListener {
            val intent = Intent(this, DownloadActivity::class.java)
            startActivity(intent)
        }

        binding.downloadWithForegroundService.setOnClickListener {
            val intent = Intent(this, DownloadForegroundActivity::class.java)
            startActivity(intent)
        }
    }
}