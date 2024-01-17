package com.waynestalk.googlepayexample

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.ActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.wallet.PaymentData
import com.google.android.gms.wallet.button.ButtonConstants.ButtonTheme
import com.google.android.gms.wallet.button.ButtonConstants.ButtonType
import com.google.android.gms.wallet.button.ButtonOptions
import com.waynestalk.googlepayexample.databinding.MainActivityBinding
import org.json.JSONException
import org.json.JSONObject

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: MainActivityBinding
    private val viewModel: MainViewModel by viewModels()

    private val resolvePaymentForResult = registerForActivityResult(
        ActivityResultContracts.StartIntentSenderForResult()
    ) { result: ActivityResult ->
        when (result.resultCode) {
            RESULT_OK -> {
                val resultData = result.data
                if (resultData != null) {
                    val paymentData = PaymentData.getFromIntent(resultData)
                    paymentData?.let {
                        handlePaymentSuccess(it)
                    }
                }
            }

            RESULT_CANCELED -> {
                // The user cancelled the payment
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initPayButton()
        initViewModel()
    }

    private fun initPayButton() {
        try {
            val buttonOptions = ButtonOptions.newBuilder()
                .setButtonTheme(ButtonTheme.LIGHT)
                .setButtonType(ButtonType.PAY)
                .setAllowedPaymentMethods(viewModel.getAllowedPaymentMethods(true).toString())
                .build()
            binding.payButton.initialize(buttonOptions)

            binding.payButton.setOnClickListener {
                viewModel.pay(900)
            }
        } catch (e: JSONException) {
            Log.e(TAG, "Error on getting payment methods", e)
        }

        binding.payButton.visibility = View.INVISIBLE
        viewModel.isGooglePayReady.observe(this) {
            if (it) {
                binding.payButton.visibility = View.VISIBLE
            } else {
                binding.payButton.visibility = View.INVISIBLE
                Toast.makeText(
                    this,
                    "Google Pay is not available on this device",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun initViewModel() {
        viewModel.paymentUi.observe(this) {
            resolvePaymentForResult.launch(IntentSenderRequest.Builder(it).build())
        }

        viewModel.paymentSuccess.observe(this) {
            handlePaymentSuccess(it)
        }

        viewModel.paymentError.observe(this) {
            Log.e(TAG, "Payment failed", it)
        }
    }

    private fun handlePaymentSuccess(paymentData: PaymentData) {
        val json = JSONObject(paymentData.toJson())
        val token = json.getJSONObject("paymentMethodData")
            .getJSONObject("tokenizationData")
            .optString("token")
        Log.d(TAG, "token = $token")
        // Send the token to the server

        startActivity(Intent(this, PaymentSuccessActivity::class.java))
    }
}