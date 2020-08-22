package com.waynestalk.jwtexample.model

import javax.persistence.*

@Entity
data class Member(
        @Column(unique = true) val username: String,
        val password: String,
        val name: String,
        @ElementCollection val authorities: Collection<String>,
        @Id @GeneratedValue var id: Long? = null
)