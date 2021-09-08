package com.waynestalk.jpacustomsql

import com.waynestalk.jpacustomsql.employee.EmployeeRepository
import org.springframework.data.domain.Pageable
import org.springframework.stereotype.Service

@Service
class EmployeeService(private val employeeRepository: EmployeeRepository) {
    fun findAll(name: String?, pageable: Pageable) = employeeRepository.findAll(name, pageable)

    fun findAllDetail(name: String?, pageable: Pageable) = employeeRepository.findAllDetailed(name, pageable)
}