package com.waynestalk.googlepayexample

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.waynestalk.googlepayexample.databinding.PaymentSuccessActivityBinding

class PaymentSuccessActivity : AppCompatActivity() {
    private lateinit var binding: PaymentSuccessActivityBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = PaymentSuccessActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)
    }
}