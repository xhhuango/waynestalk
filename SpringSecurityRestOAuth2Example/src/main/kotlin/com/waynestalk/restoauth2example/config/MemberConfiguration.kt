package com.waynestalk.restoauth2example.config

import com.waynestalk.restoauth2example.model.Member
import com.waynestalk.restoauth2example.repository.MemberRepository
import org.springframework.boot.ApplicationRunner
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class MemberConfiguration {
    @Bean
    fun initMembers(memberRepository: MemberRepository) = ApplicationRunner {
        memberRepository.saveAll(listOf(
                Member("monika@gmail.com", "google", "Monika", listOf("ROLE_ADMIN", "ROLE_USER")),
                Member("jack@gmail.com", "google", "Jack", listOf("ROLE_USER")),
                Member("peter@gmail.com", "google", "Peter", listOf("ROLE_USER"))
        ))
    }
}