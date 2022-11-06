package com.waynestalk.broadcastreceiverexample

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

class LocaleChangedBroadcastReceiver : BroadcastReceiver() {
    companion object {
        private const val TAG = "LocaleChangedBroadcast"
    }

    override fun onReceive(context: Context, intent: Intent) {
        Log.d(TAG, "Received: locale changed => ${Thread.currentThread()}")
    }
}