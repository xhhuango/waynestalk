package com.waynestalk.example

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

interface Service {
    data class PostRequest(
        val name: String,
        val sex: Int,
    )

    data class PostResponse(
        val data: PostRequest,
        val json: PostRequest,
        val headers: Map<String, String>,
        val url: String,
    )

    @POST("/post")
    suspend fun post(@Body request: PostRequest): Response<PostResponse>
}