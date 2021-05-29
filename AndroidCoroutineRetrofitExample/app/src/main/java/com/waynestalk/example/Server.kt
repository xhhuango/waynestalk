package com.waynestalk.example

import android.util.Log
import kotlinx.coroutines.Dispatchers
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import kotlinx.coroutines.withContext

object Server {
    private const val URL = "https://postman-echo.com"

    private val tag = Server::class.java.name

    private val service: Service

    init {
        val client = OkHttpClient.Builder().build()
        val retrofit = Retrofit.Builder()
            .baseUrl(URL)
            .addConverterFactory(GsonConverterFactory.create())
            .client(client)
            .build()
        service = retrofit.create(Service::class.java)
    }

    suspend fun post(name: String, sex: Int): Pair<String, Int> = withContext(Dispatchers.IO) {
        Log.d(tag, "Thread is ${Thread.currentThread().name}")

        val request = Service.PostRequest(name, sex)
        val response = service.post(request)
        if (response.isSuccessful) {
            val body = response.body()!!
            return@withContext Pair(body.json.name, body.json.sex)
        } else {
            throw Exception(response.errorBody()?.charStream()?.readText())
        }
    }
}