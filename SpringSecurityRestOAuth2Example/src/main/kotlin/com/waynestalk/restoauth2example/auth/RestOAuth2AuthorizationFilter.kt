package com.waynestalk.restoauth2example.auth

import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.stereotype.Component
import org.springframework.util.StringUtils
import org.springframework.web.filter.GenericFilterBean
import javax.servlet.FilterChain
import javax.servlet.ServletRequest
import javax.servlet.ServletResponse
import javax.servlet.http.HttpServletRequest

@Component
class RestOAuth2AuthorizationFilter(private val tokenManager: TokenManager) : GenericFilterBean() {
    companion object {
        private const val authenticationHeader = "Authorization"
        private const val authenticationScheme = "Bearer"
    }

    override fun doFilter(request: ServletRequest, response: ServletResponse, chain: FilterChain) {
        val token = extractToken(request as HttpServletRequest)
        if (token != null && StringUtils.hasText(token)) {
            val authentication = tokenManager[token]
            if (authentication != null) {
                SecurityContextHolder.getContext().authentication = authentication
            }
        }

        chain.doFilter(request, response)
    }

    private fun extractToken(request: HttpServletRequest): String? {
        val bearerToken = request.getHeader(authenticationHeader)
        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith("$authenticationScheme ")) {
            return bearerToken.substring(authenticationScheme.length + 1)
        }
        return null
    }
}