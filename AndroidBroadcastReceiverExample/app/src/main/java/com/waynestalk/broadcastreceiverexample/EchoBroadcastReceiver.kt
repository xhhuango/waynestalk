package com.waynestalk.broadcastreceiverexample

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.widget.Toast

class EchoBroadcastReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != "com.waynestalk.echo") return

        val message = intent.getStringExtra("message")
        Toast.makeText(context, "Received: $message", Toast.LENGTH_LONG).show()
    }
}