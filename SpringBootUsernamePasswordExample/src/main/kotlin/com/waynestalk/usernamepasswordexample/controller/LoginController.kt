package com.waynestalk.usernamepasswordexample.controller

import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController

@RestController
class LoginController {
    @PostMapping("/login", consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE])
    fun login(@RequestBody request: LoginRequest) {
        throw NotImplementedError("/login should not be called")
    }

    data class LoginRequest(val username: String, val password: String)
}