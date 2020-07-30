package com.waynestalk.springdoc

import io.swagger.v3.oas.annotations.media.Schema

data class User(
        @field:Schema(description = "User name")
        val name: String,
        @field:Schema(description = "User age")
        val age: Int,
        @field:Schema(description = "User email")
        val email: String)