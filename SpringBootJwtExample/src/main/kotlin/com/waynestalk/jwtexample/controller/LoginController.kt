package com.waynestalk.jwtexample.controller

import com.waynestalk.jwtexample.auth.JwtFilter
import com.waynestalk.jwtexample.auth.JwtTokenProvider
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController
import javax.servlet.http.HttpServletResponse
import javax.validation.Valid
import javax.validation.constraints.NotBlank

@RestController
class LoginController(val jwtTokenProvider: JwtTokenProvider,
                      val authenticationManagerBuilder: AuthenticationManagerBuilder) {
    @PostMapping("/login")
    fun login(@Valid @RequestBody request: LoginRequest, httpServletResponse: HttpServletResponse): LoginResponse {
        val authenticationToken = UsernamePasswordAuthenticationToken(request.username, request.password)
        val authentication = authenticationManagerBuilder.`object`.authenticate(authenticationToken)
        SecurityContextHolder.getContext().authentication = authentication

        val token = jwtTokenProvider.generate(authentication)
        httpServletResponse.addHeader(JwtFilter.authenticationHeader, "${JwtFilter.authenticationScheme} $token")

        return LoginResponse(token)
    }

    data class LoginRequest(@field:NotBlank val username: String,
                            @field:NotBlank val password: String)

    data class LoginResponse(val token: String)
}