package com.waynestalk.demo.domain

import javax.persistence.Column
import javax.persistence.Entity
import javax.persistence.GeneratedValue
import javax.persistence.Id

@Entity
data class User(@Column(unique = true) val name: String?,
                var age: Int?,
                @Id @GeneratedValue var id: Long? = null)