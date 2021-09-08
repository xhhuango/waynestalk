package com.waynestalk.jpacustomsql.employer

import javax.persistence.Entity
import javax.persistence.GeneratedValue
import javax.persistence.GenerationType
import javax.persistence.Id

@Entity
data class Employer(
    val name: String?,
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) var employerId: Long? = null,
)