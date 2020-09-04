package com.waynestalk.restoauth2example.auth

import io.jsonwebtoken.Jwts
import io.jsonwebtoken.SignatureAlgorithm
import io.jsonwebtoken.io.Decoders
import io.jsonwebtoken.security.Keys
import org.springframework.cache.CacheManager
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken
import org.springframework.stereotype.Component
import java.util.*

@Component
class TokenManager(cacheManager: CacheManager) {
    companion object {
        private const val claimAuthorities = "authorities"
        private const val claimName = "name"
        private const val claimEmail = "email"

        private const val secret = "qsbWaaBHBN/I7FYOrev4yQFJm60sgZkWIEDlGtsRl7El/k+DbUmg8nmWiVvEfhZ91Y67Sc6Ifobi05b/XDwBy4kXUcKTitNqocy7rQ9Z3kMipYjbL3WZUJU2luigIRxhTVNw8FXdT5q56VfY0LcQv3mEp6iFm1JG43WyvGFV3hCkhLPBJV0TWnEi69CfqbUMAIjmymhGjcbqEK8Wt10bbfxkM5uar3tpyqzp3Q=="
        private val key = Keys.hmacShaKeyFor(Decoders.BASE64.decode(secret))
    }

    private val cache = cacheManager.getCache("tokenManager")!!

    operator fun get(token: String): OAuth2AuthenticationToken? {
        return if (validate(token)) {
            val authentication = cache.get(token)?.get()
            authentication as OAuth2AuthenticationToken
        } else {
            cache.evict(token)
            null
        }
    }

    operator fun set(token: String, authentication: OAuth2AuthenticationToken) {
        cache.put(token, authentication)
    }

    fun generate(authentication: OAuth2AuthenticationToken): String {
        val subject = authentication.name
        val name = authentication.principal.attributes["name"]
        val email = authentication.principal.attributes["email"]
        val authorities = authentication.authorities?.joinToString { it.authority } ?: ""
        val expiration = Date(System.currentTimeMillis() + (60 * 60 * 1000))
        return Jwts.builder()
                .setSubject(subject)
                .claim(claimAuthorities, authorities)
                .claim(claimName, name)
                .claim(claimEmail, email)
                .signWith(key, SignatureAlgorithm.HS512)
                .setExpiration(expiration)
                .compact()
    }

    private fun validate(token: String): Boolean {
        return try {
            val jwtParser = Jwts.parserBuilder().setSigningKey(key).build()
            jwtParser.parse(token)
            true
        } catch (e: Exception) {
            false
        }
    }
}