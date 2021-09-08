package com.waynestalk.jpacustomsql

import com.waynestalk.jpacustomsql.employee.Employee
import com.waynestalk.jpacustomsql.employee.EmployeeRepository
import com.waynestalk.jpacustomsql.employer.Employer
import com.waynestalk.jpacustomsql.employer.EmployerRepository
import org.springframework.boot.ApplicationRunner
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.util.*

@Configuration
class DatabaseInitialization(
    private val employerRepository: EmployerRepository,
    private val employeeRepository: EmployeeRepository,
) {
    @Bean
    fun initData() = ApplicationRunner {
        val employer1 = Employer("Employer 1")
        val employer2 = Employer("Employer 2")
        employerRepository.save(employer1)
        employerRepository.save(employer2)

        for (i in 1..20) {
            val employerId = if (i % 2 == 0) employer1.employerId else employer2.employerId
            val employee = Employee(
                "Employee $i",
                employerId!!,
                Date(System.currentTimeMillis() + i * 1000),
            )
            employeeRepository.save(employee)
        }
    }
}