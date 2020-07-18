package com.waynestalk.demo.repository

import com.waynestalk.demo.domain.User
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository

@Repository
interface UserRepository : JpaRepository<User, Long> {
    fun findAllByAge(age: Int): List<User>

    fun findByName(name: String): User?
}