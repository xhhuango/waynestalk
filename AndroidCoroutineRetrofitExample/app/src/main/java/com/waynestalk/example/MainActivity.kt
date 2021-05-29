package com.waynestalk.example

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private val tag = MainActivity::class.java.name

    private lateinit var postButton: Button
    private lateinit var nameTextView: TextView
    private lateinit var ageTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        postButton = findViewById(R.id.postButton)
        nameTextView = findViewById(R.id.nameTextView)
        ageTextView = findViewById(R.id.ageTextView)

        postButton.setOnClickListener {
            CoroutineScope(Dispatchers.Main).launch {
                val (name, sex) = Server.post("Wayne", 1)
                Log.d(tag, "Thread is ${Thread.currentThread().name}")
                nameTextView.text = name
                ageTextView.text = if (sex == 1) "male" else "female"
            }
        }
    }
}