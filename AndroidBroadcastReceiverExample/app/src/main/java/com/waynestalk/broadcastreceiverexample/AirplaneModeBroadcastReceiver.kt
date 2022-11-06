package com.waynestalk.broadcastreceiverexample

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.widget.Toast

class AirplaneModeBroadcastReceiver : BroadcastReceiver() {
    companion object {
        private const val TAG = "AirplaneModeBroadcast"
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_AIRPLANE_MODE_CHANGED) return

        val isOn = intent.getBooleanExtra("state", false)
        val message = "Airplane mode is ${if (isOn) "on" else "off"}."
        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
    }
}