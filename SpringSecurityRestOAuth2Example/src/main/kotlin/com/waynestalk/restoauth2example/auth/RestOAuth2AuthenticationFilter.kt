package com.waynestalk.restoauth2example.auth

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.waynestalk.restoauth2example.model.Member
import com.waynestalk.restoauth2example.repository.MemberRepository
import org.springframework.http.HttpMethod
import org.springframework.http.MediaType
import org.springframework.security.authentication.AuthenticationManager
import org.springframework.security.authentication.ProviderManager
import org.springframework.security.core.GrantedAuthority
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.core.userdetails.UserDetails
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken
import org.springframework.security.oauth2.client.authentication.OAuth2LoginAuthenticationToken
import org.springframework.security.oauth2.client.endpoint.DefaultAuthorizationCodeTokenResponseClient
import org.springframework.security.oauth2.client.oidc.authentication.OidcAuthorizationCodeAuthenticationProvider
import org.springframework.security.oauth2.client.oidc.userinfo.OidcUserService
import org.springframework.security.oauth2.client.registration.ClientRegistrationRepository
import org.springframework.security.oauth2.client.web.DefaultOAuth2AuthorizationRequestResolver
import org.springframework.security.oauth2.client.web.OAuth2AuthorizedClientRepository
import org.springframework.security.oauth2.core.OAuth2AuthenticationException
import org.springframework.security.oauth2.core.OAuth2Error
import org.springframework.security.oauth2.core.endpoint.OAuth2AuthorizationExchange
import org.springframework.security.oauth2.core.endpoint.OAuth2AuthorizationResponse
import org.springframework.security.web.util.matcher.AntPathRequestMatcher
import org.springframework.stereotype.Component
import org.springframework.web.filter.GenericFilterBean
import javax.servlet.FilterChain
import javax.servlet.ServletRequest
import javax.servlet.ServletResponse
import javax.servlet.http.HttpServletRequest
import javax.servlet.http.HttpServletResponse

@Component
class RestOAuth2AuthenticationFilter(
        private val clientRegistrationRepository: ClientRegistrationRepository,
        private val authorizedClientRepository: OAuth2AuthorizedClientRepository,
        private val tokenManager: TokenManager,
        private val userDetailsService: MemberUserDetailsService,
        private val memberRepository: MemberRepository
) : GenericFilterBean() {
    companion object {
        private const val contentTypeHeader = "Content-Type"
        private const val baseUri = "/auth"
        private const val registrationIdUriVariableName = "registrationId"
        private const val redirectUri = "postmessage"
        private const val nonceParameterName = "nonce"
    }

    private val authenticationManager: AuthenticationManager
    private val authorizationRequestResolver = DefaultOAuth2AuthorizationRequestResolver(clientRegistrationRepository, baseUri)
    private val requestMatcher = AntPathRequestMatcher("$baseUri/{$registrationIdUriVariableName}", HttpMethod.POST.name)

    init {
        val accessTokenResponseClient = DefaultAuthorizationCodeTokenResponseClient()
        val userService = OidcUserService()
        val authenticationProvider = OidcAuthorizationCodeAuthenticationProvider(accessTokenResponseClient, userService)
        authenticationManager = ProviderManager(authenticationProvider)

        authorizationRequestResolver.setAuthorizationRequestCustomizer {
            it.redirectUri(redirectUri)
            it.additionalParameters { additionalParameters -> additionalParameters.remove(nonceParameterName) }
            it.attributes { attributes -> attributes.remove(nonceParameterName) }
        }
    }

    override fun doFilter(req: ServletRequest, res: ServletResponse, chain: FilterChain) {
        val request = req as HttpServletRequest
        val response = res as HttpServletResponse

        if (!requireAuthentication(request)) {
            chain.doFilter(request, response)
            return
        }

        try {
            val authentication = authenticate(request, response)
            successfulAuthentication(response, authentication)
        } catch (e: Exception) {
            unsuccessfulAuthentication(response, e)
        }
    }

    private fun requireAuthentication(request: HttpServletRequest) = requestMatcher.matches(request)

    private fun authenticate(request: HttpServletRequest, response: HttpServletResponse): OAuth2AuthenticationToken {
        val code = readCode(request)
                ?: throw OAuth2AuthenticationException(OAuth2Error("authentication_code_missing"))
        val registrationId = requestMatcher.matcher(request).variables[registrationIdUriVariableName]
                ?: throw OAuth2AuthenticationException(OAuth2Error("client_registration_not_found"))
        val clientRegistration = clientRegistrationRepository.findByRegistrationId(registrationId)
                ?: throw OAuth2AuthenticationException(OAuth2Error("client_registration_not_found"))

        val authorizationRequest = authorizationRequestResolver.resolve(request, registrationId)

        val authorizationResponse = OAuth2AuthorizationResponse
                .success(code)
                .redirectUri(redirectUri)
                .state(authorizationRequest.state)
                .build()

        val authenticationRequest = OAuth2LoginAuthenticationToken(
                clientRegistration, OAuth2AuthorizationExchange(authorizationRequest, authorizationResponse))

        val authenticationResult = authenticationManager.authenticate(authenticationRequest)
                as OAuth2LoginAuthenticationToken

        val username = authenticationResult.principal.attributes["email"] as String
        val user = loadUser(username) ?: createUser(authenticationResult)
        val authorities = mergeAuthorities(authenticationResult, user)

        val oauth2Authentication = OAuth2AuthenticationToken(
                authenticationResult.principal,
                authorities,
                authenticationResult.clientRegistration.registrationId)

        val authorizedClient = OAuth2AuthorizedClient(
                authenticationResult.clientRegistration,
                oauth2Authentication.name,
                authenticationResult.accessToken,
                authenticationResult.refreshToken)

        authorizedClientRepository.saveAuthorizedClient(authorizedClient, oauth2Authentication, request, response)

        return oauth2Authentication
    }

    private fun readCode(request: HttpServletRequest): String? {
        val authRequest: Map<String, Any> = jacksonObjectMapper().readValue(request.reader)
        return authRequest["code"] as? String
    }

    private fun loadUser(username: String): UserDetails? {
        return try {
            userDetailsService.loadUserByUsername(username)
        } catch (e: Exception) {
            null
        }
    }

    private fun createUser(authentication: OAuth2LoginAuthenticationToken): UserDetails {
        val attributes = authentication.principal.attributes
        val username = attributes["email"] as String
        val member = Member(
                username,
                authentication.clientRegistration.registrationId,
                attributes["name"] as String,
                listOf("ROLE_USER")
        )
        memberRepository.save(member)

        return userDetailsService.loadUserByUsername(username)
    }

    private fun mergeAuthorities(authentication: OAuth2LoginAuthenticationToken, user: UserDetails): Collection<GrantedAuthority> {
        val authorities = HashSet<GrantedAuthority>()
        authorities.addAll(authentication.authorities)
        authorities.addAll(user.authorities)
        return authorities
    }

    private fun successfulAuthentication(response: HttpServletResponse, authentication: OAuth2AuthenticationToken) {
        SecurityContextHolder.getContext().authentication = authentication

        val token = tokenManager.generate(authentication)
        tokenManager[token] = authentication

        response.addHeader(contentTypeHeader, MediaType.APPLICATION_JSON_VALUE)
        response.writer.println(jacksonObjectMapper().writeValueAsString(TokenResponse(token)))
    }

    private fun unsuccessfulAuthentication(response: HttpServletResponse, exception: Exception) {
        SecurityContextHolder.clearContext()
        response.sendError(HttpServletResponse.SC_UNAUTHORIZED, exception.message)
    }

    private data class TokenResponse(val token: String)
}