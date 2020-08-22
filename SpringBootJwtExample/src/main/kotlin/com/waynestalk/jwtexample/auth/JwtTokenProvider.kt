package com.waynestalk.jwtexample.auth

import io.jsonwebtoken.Jwts
import io.jsonwebtoken.SignatureAlgorithm
import io.jsonwebtoken.io.Decoders
import io.jsonwebtoken.security.Keys
import org.slf4j.LoggerFactory
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.Authentication
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.userdetails.User
import org.springframework.stereotype.Component
import java.util.*

@Component
class JwtTokenProvider {
    companion object {
        const val claimAuthorities = "authorities"
    }

    private val logger = LoggerFactory.getLogger(this::class.java)
    private val secret = "qsbWaaBHBN/I7FYOrev4yQFJm60sgZkWIEDlGtsRl7El/k+DbUmg8nmWiVvEfhZ91Y67Sc6Ifobi05b/XDwBy4kXUcKTitNqocy7rQ9Z3kMipYjbL3WZUJU2luigIRxhTVNw8FXdT5q56VfY0LcQv3mEp6iFm1JG43WyvGFV3hCkhLPBJV0TWnEi69CfqbUMAIjmymhGjcbqEK8Wt10bbfxkM5uar3tpyqzp3Q=="
    private val key = Keys.hmacShaKeyFor(Decoders.BASE64.decode(secret))

    fun generate(authentication: Authentication): String {
        val authorities = authentication.authorities?.joinToString { it.authority } ?: ""
        val expiration = Date(System.currentTimeMillis() + (60 * 60 * 1000))
        return Jwts.builder()
                .setSubject(authentication.name)
                .claim(claimAuthorities, authorities)
                .signWith(key, SignatureAlgorithm.HS512)
                .setExpiration(expiration)
                .compact()
    }

    fun toAuthentication(token: String): Authentication {
        val jwtParser = Jwts.parserBuilder().setSigningKey(key).build()
        val claims = jwtParser.parseClaimsJws(token).body
        val authorities = claims[claimAuthorities].toString().split(",").map { SimpleGrantedAuthority(it) }
        val user = User(claims.subject, "", authorities)
        return UsernamePasswordAuthenticationToken(user, token, authorities)
    }

    fun validate(token: String): Boolean {
        val jwtParser = Jwts.parserBuilder().setSigningKey(key).build()

        try {
            jwtParser.parse(token)
            return true
        } catch (e: Exception) {
            logger.error(e.message)
        }

        return false
    }
}