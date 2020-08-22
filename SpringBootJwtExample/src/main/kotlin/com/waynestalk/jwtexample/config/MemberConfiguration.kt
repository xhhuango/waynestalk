package com.waynestalk.jwtexample.config

import com.waynestalk.jwtexample.repository.MemberRepository
import com.waynestalk.jwtexample.model.Member
import org.springframework.boot.ApplicationRunner
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.security.crypto.password.PasswordEncoder

@Configuration
class MemberConfiguration {
    @Bean
    fun initMembers(memberRepository: MemberRepository, passwordEncoder: PasswordEncoder) = ApplicationRunner {
        memberRepository.saveAll(listOf(
                Member("monika", passwordEncoder.encode("123456"), "Monika", listOf("ROLE_ADMIN", "ROLE_USER")),
                Member("jack", passwordEncoder.encode("123456"), "Jack", listOf("ROLE_USER")),
                Member("peter", "123456", "Peter", listOf("ROLE_USER"))
        ))
    }
}