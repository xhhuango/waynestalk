package com.waynestalk.jwtexample.repository

import com.waynestalk.jwtexample.model.Member
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query

interface MemberRepository : JpaRepository<Member, Long> {
    @Query("SELECT m FROM Member m JOIN FETCH m.authorities WHERE m.username = (:username)")
    fun findByUsername(username: String): Member?
}