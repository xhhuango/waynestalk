package com.waynestalk.jwtexample.auth

import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.util.StringUtils
import org.springframework.web.filter.GenericFilterBean
import javax.servlet.FilterChain
import javax.servlet.ServletRequest
import javax.servlet.ServletResponse
import javax.servlet.http.HttpServletRequest

class JwtFilter(private val jwtTokenProvider: JwtTokenProvider) : GenericFilterBean() {
    companion object {
        const val authenticationHeader = "Authorization"
        const val authenticationScheme = "Bearer"
    }

    override fun doFilter(request: ServletRequest, response: ServletResponse, chain: FilterChain) {
        val token = extractToken(request as HttpServletRequest)
        if (StringUtils.hasText(token) && jwtTokenProvider.validate(token)) {
            SecurityContextHolder.getContext().authentication = jwtTokenProvider.toAuthentication(token)
        }

        chain.doFilter(request, response)
    }

    private fun extractToken(request: HttpServletRequest): String {
        val bearerToken = request.getHeader(authenticationHeader)
        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith("$authenticationScheme ")) {
            return bearerToken.substring(authenticationScheme.length + 1)
        }
        return ""
    }
}