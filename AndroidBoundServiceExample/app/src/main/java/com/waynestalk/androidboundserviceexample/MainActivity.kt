package com.waynestalk.androidboundserviceexample

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.waynestalk.androidboundserviceexample.databinding.MainActivityBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: MainActivityBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.downloadWithBoundService.setOnClickListener {
            val intent = Intent(this, DownloadBoundActivity::class.java)
            startActivity(intent)
        }

        binding.downloadWithRemoteBoundService.setOnClickListener {
            val intent = Intent(this, DownloadRemoteBoundActivity::class.java)
            startActivity(intent)
        }
    }
}