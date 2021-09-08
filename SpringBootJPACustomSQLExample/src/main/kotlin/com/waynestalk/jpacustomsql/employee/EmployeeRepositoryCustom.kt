package com.waynestalk.jpacustomsql.employee

import com.waynestalk.jpacustomsql.employer.Employer
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable

interface EmployeeRepositoryCustom {
    fun findAll(name: String?, pageable: Pageable): Page<Employee>

    fun findAllDetailed(name: String?, pageable: Pageable): Page<Pair<Employee, Employer>>
}