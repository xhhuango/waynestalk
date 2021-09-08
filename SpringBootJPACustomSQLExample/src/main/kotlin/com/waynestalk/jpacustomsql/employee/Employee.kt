package com.waynestalk.jpacustomsql.employee

import java.util.*
import javax.persistence.Entity
import javax.persistence.GeneratedValue
import javax.persistence.GenerationType
import javax.persistence.Id

@Entity
data class Employee(
    val name: String,
    val employerId: Long,
    val createdAt: Date? = null,
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) var employeeId: Long? = null,
)
