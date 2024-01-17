package com.waynestalk.googlepayexample

import android.app.Application
import android.app.PendingIntent
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.google.android.gms.common.api.ApiException
import com.google.android.gms.common.api.ResolvableApiException
import com.google.android.gms.tasks.Task
import com.google.android.gms.wallet.IsReadyToPayRequest
import com.google.android.gms.wallet.PaymentData
import com.google.android.gms.wallet.PaymentDataRequest
import com.google.android.gms.wallet.PaymentsClient
import com.google.android.gms.wallet.Wallet
import com.google.android.gms.wallet.WalletConstants
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import org.json.JSONArray
import org.json.JSONObject
import java.math.BigDecimal
import java.math.RoundingMode

class MainViewModel(application: Application) : AndroidViewModel(application) {
    companion object {
        private const val TAG = "MainViewModel"
    }

    val isGooglePayReady: MutableLiveData<Boolean> = MutableLiveData()
    val paymentSuccess: MutableLiveData<PaymentData> = MutableLiveData()
    val paymentError: MutableLiveData<Exception?> = MutableLiveData()
    val paymentUi: MutableLiveData<PendingIntent> = MutableLiveData()

    private val paymentsClient: PaymentsClient

    init {
        val walletOptions = Wallet.WalletOptions.Builder()
            .setEnvironment(WalletConstants.ENVIRONMENT_TEST)
            .build()
        paymentsClient = Wallet.getPaymentsClient(application, walletOptions)

        viewModelScope.launch {
            checkIfGooglePayIsReady()
        }
    }

    private suspend fun checkIfGooglePayIsReady() {
        try {
            val jsonRequest = JSONObject().apply {
                put("apiVersion", 2)
                put("apiVersionMinor", 0)
                put("allowedPaymentMethods", getAllowedPaymentMethods(false))
            }
            val request = IsReadyToPayRequest.fromJson(jsonRequest.toString())
            Log.d(TAG, jsonRequest.toString())
            val task = paymentsClient.isReadyToPay(request)
            if (task.await()) {
                isGooglePayReady.postValue(true)
            } else {
                Log.e(TAG, "Error on requesting if Google Pay is ready")
                isGooglePayReady.postValue(false)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error on checking if Google Pay is ready", e)
            isGooglePayReady.postValue(false)
        }
    }

    fun getAllowedPaymentMethods(withToken: Boolean): JSONArray {
        val jsonObject = getCardPaymentMethod(withToken)
        return JSONArray().put(jsonObject)
    }

    private fun getCardPaymentMethod(withToken: Boolean): JSONObject {
        return JSONObject().apply {
            put("type", "CARD")

            put("parameters", JSONObject().apply {
                put("allowedAuthMethods", JSONArray().apply {
                    put("PAN_ONLY")
                    put("CRYPTOGRAM_3DS")
                })

                put("allowedCardNetworks", JSONArray().apply {
                    put("AMEX")
                    put("DISCOVER")
                    put("JCB")
                    put("MASTERCARD")
                    put("VISA")
                })

                put("billingAddressRequired", true)
                put("billingAddressParameters", JSONObject().apply {
                    put("format", "FULL")
                })
            })

            if (withToken) {
                put("tokenizationSpecification", JSONObject().apply {
                    put("type", "PAYMENT_GATEWAY")
                    put("parameters", JSONObject().apply {
                        put("gateway", "example")
                        put("gatewayMerchantId", "exampleGatewayMerchantId")
                    })
                })
            }
        }
    }

    fun pay(priceCents: Long) {
        getPaymentDataRequest(priceCents).addOnCompleteListener { task ->
            if (task.isSuccessful) {
                paymentSuccess.postValue(task.result)
            } else {
                when (val e = task.exception) {
                    is ResolvableApiException -> paymentUi.postValue(e.resolution)
                    is ApiException -> paymentError.postValue(e)
                    else -> paymentError.postValue(e)
                }
            }
        }
    }

    private fun getPaymentDataRequest(priceCents: Long): Task<PaymentData> {
        // ex: 9.5
        val price = BigDecimal(priceCents)
            .divide(BigDecimal(100))
            .setScale(2, RoundingMode.HALF_EVEN)
            .toString()

        val jsonRequest = JSONObject().apply {
            put("apiVersion", 2)
            put("apiVersionMinor", 0)
            put("allowedPaymentMethods", getAllowedPaymentMethods(true))
            put("transactionInfo", JSONObject().apply {
                put("totalPrice", price)
                put("totalPriceStatus", "FINAL")
                put("countryCode", "US")
                put("currencyCode", "USD")
            })
            put("merchantInfo", JSONObject().put("merchantName", "Example Merchant"))

            put("shippingAddressParameters", JSONObject().apply {
                put("phoneNumberRequired", false)
                put("allowedCountryCodes", JSONArray(listOf("US", "GB")))
            })
            put("shippingAddressRequired", true)
        }

        val request = PaymentDataRequest.fromJson(jsonRequest.toString())
        Log.d(TAG, jsonRequest.toString())
        return paymentsClient.loadPaymentData(request)
    }
}