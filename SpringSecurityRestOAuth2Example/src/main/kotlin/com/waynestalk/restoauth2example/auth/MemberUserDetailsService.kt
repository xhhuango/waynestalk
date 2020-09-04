package com.waynestalk.restoauth2example.auth

import com.waynestalk.restoauth2example.repository.MemberRepository
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.userdetails.User
import org.springframework.security.core.userdetails.UserDetails
import org.springframework.security.core.userdetails.UserDetailsService
import org.springframework.security.core.userdetails.UsernameNotFoundException
import org.springframework.stereotype.Component

@Component
class MemberUserDetailsService(private val memberRepository: MemberRepository) : UserDetailsService {
    override fun loadUserByUsername(username: String): UserDetails {
        val member = memberRepository.findByUsername(username)
                ?: throw UsernameNotFoundException("$username was not found")

        val authority = member.authorities.map { SimpleGrantedAuthority(it) }
        return User(member.username, "", authority)
    }
}