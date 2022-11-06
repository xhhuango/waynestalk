package com.waynestalk.broadcastreceiverexample

import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.waynestalk.broadcastreceiverexample.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    private val broadcastReceiver = AirplaneModeBroadcastReceiver()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.echoButton.setOnClickListener {
            val intent = Intent().apply {
                action = "com.waynestalk.echo"
                `package` = "com.waynestalk.broadcastreceiverexample"
                putExtra("message", "Hello Android")
            }
            sendBroadcast(intent)
        }
    }

    override fun onResume() {
        super.onResume()

        val filter = IntentFilter(Intent.ACTION_AIRPLANE_MODE_CHANGED)
        val flags = ContextCompat.RECEIVER_EXPORTED
        ContextCompat.registerReceiver(this, broadcastReceiver, filter, flags)
    }

    override fun onPause() {
        super.onPause()

        unregisterReceiver(broadcastReceiver)
    }
}