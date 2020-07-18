package com.waynestalk.demo

import com.waynestalk.demo.domain.User
import com.waynestalk.demo.repository.UserRepository
import org.springframework.boot.ApplicationRunner
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class UserConfiguration {
    @Bean
    fun initUsers(userRepository: UserRepository) = ApplicationRunner {
        userRepository.saveAll(listOf(
                User("Jason", 20, 1),
                User("Alan", 22, 2),
                User("David", 21, 3),
                User("Monika", 20, 4),
                User("Angela", 22, 5)
        ))
    }
}
