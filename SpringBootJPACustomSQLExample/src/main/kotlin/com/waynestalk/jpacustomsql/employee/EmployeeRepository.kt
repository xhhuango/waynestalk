package com.waynestalk.jpacustomsql.employee

import org.springframework.data.jpa.repository.JpaRepository

interface EmployeeRepository: JpaRepository<Employee, Long>, EmployeeRepositoryCustom {
    fun findByEmployeeId(employeeId: Long): Employee?
}