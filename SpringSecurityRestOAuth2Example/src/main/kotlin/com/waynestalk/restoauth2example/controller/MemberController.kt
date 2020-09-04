package com.waynestalk.restoauth2example.controller

import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken
import org.springframework.security.oauth2.client.web.OAuth2AuthorizedClientRepository
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController
import java.security.Principal
import javax.servlet.http.HttpServletRequest

@RestController
class MemberController(private val authorizedClientRepository: OAuth2AuthorizedClientRepository) {
    @GetMapping("/user")
    @PreAuthorize("hasRole('ROLE_USER')")
    fun user(principal: Principal): UserResponse {
        val authentication = principal as OAuth2AuthenticationToken
        return UserResponse(
                authentication.principal.attributes["email"] as String,
                authentication.principal.attributes["name"] as String
        )
    }

    data class UserResponse(val email: String, val name: String)

    @GetMapping("/admin")
    @PreAuthorize("hasRole('ROLE_ADMIN')")
    fun admin(principal: Principal, request: HttpServletRequest): AdminResponse {
        val authentication = principal as OAuth2AuthenticationToken
        val authorizedClient = authorizedClientRepository.loadAuthorizedClient<OAuth2AuthorizedClient>(
                authentication.authorizedClientRegistrationId,
                authentication,
                request)
        return AdminResponse(
                authentication.principal.attributes["email"] as String,
                authentication.principal.attributes["name"] as String,
                authorizedClient.accessToken.tokenValue,
                authorizedClient.refreshToken?.tokenValue ?: ""
        )
    }

    data class AdminResponse(val email: String, val name: String, val accessToken: String, val refreshToken: String)
}