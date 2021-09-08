package com.waynestalk.jpacustomsql

import com.waynestalk.jpacustomsql.employee.Employee
import com.waynestalk.jpacustomsql.employer.Employer
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.domain.Sort
import org.springframework.data.web.SortDefault
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/employees")
class EmployeeController(private val employeeService: EmployeeService) {
    @GetMapping
    fun list(
        @RequestParam(required = false) name: String?,
        @SortDefault(sort = ["createdAt"], direction = Sort.Direction.DESC) pageable: Pageable,
    ): Page<Employee> = employeeService.findAll(name, pageable)

    @GetMapping("/detail")
    fun listDetail(
        @RequestParam(required = false) name: String?,
        @SortDefault(sort = ["createdAt"], direction = Sort.Direction.DESC) pageable: Pageable,
    ): Page<Pair<Employee, Employer>> = employeeService.findAllDetail(name, pageable)
}