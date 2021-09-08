package com.waynestalk.jpacustomsql.employee

import com.waynestalk.jpacustomsql.employer.Employer
import org.springframework.data.domain.Page
import org.springframework.data.domain.PageImpl
import org.springframework.data.domain.Pageable
import javax.persistence.EntityManager
import javax.persistence.PersistenceContext

class EmployeeRepositoryCustomImpl(
    @PersistenceContext private val entityManager: EntityManager,
) : EmployeeRepositoryCustom {
    override fun findAll(name: String?, pageable: Pageable): Page<Employee> {
        val sb = StringBuilder()
        sb.append("FROM Employee e")

        val where = mutableListOf<String>()
        val parameters = mutableMapOf<String, Any>()
        if (!name.isNullOrBlank()) {
            where.add("e.name LIKE CONCAT('%', :name, '%')")
            parameters["name"] = name
        }

        val countQuery = entityManager.createQuery("SELECT COUNT(1) $sb", Long::class.javaObjectType)
        parameters.forEach { countQuery.setParameter(it.key, it.value) }
        val count = countQuery.singleResult

        if (!pageable.sort.isEmpty) {
            val sorts = pageable.sort.map { "e.${it.property} ${if (it.isAscending) "ASC" else "DESC"}" }
            sb.append(" ORDER BY ${sorts.joinToString(", ")}")
        }
        val listQuery = entityManager.createQuery("SELECT e AS employee $sb")
        parameters.forEach { listQuery.setParameter(it.key, it.value) }
        listQuery.maxResults = pageable.pageSize
        listQuery.firstResult = pageable.offset.toInt()
        val list = listQuery.resultList.map {
            return@map it as Employee
        }

        return PageImpl(list, pageable, count)
    }

    override fun findAllDetailed(name: String?, pageable: Pageable): Page<Pair<Employee, Employer>> {
        val sb = StringBuilder()
        sb.append("FROM Employee e")
        sb.append(" LEFT JOIN Employer er ON er.employerId = e.employerId")

        val where = mutableListOf<String>()
        val parameters = mutableMapOf<String, Any>()
        if (!name.isNullOrBlank()) {
            where.add("e.name LIKE CONCAT('%', :name, '%')")
            parameters["name"] = name
        }

        val countQuery = entityManager.createQuery("SELECT COUNT(1) $sb", Long::class.javaObjectType)
        parameters.forEach { countQuery.setParameter(it.key, it.value) }
        val count = countQuery.singleResult

        if (!pageable.sort.isEmpty) {
            val sorts = pageable.sort.map { "e.${it.property} ${if (it.isAscending) "ASC" else "DESC"}" }
            sb.append(" ORDER BY ${sorts.joinToString(", ")}")
        }
        val listQuery = entityManager.createQuery("SELECT e AS employee, er AS employer $sb")
        parameters.forEach { listQuery.setParameter(it.key, it.value) }
        listQuery.maxResults = pageable.pageSize
        listQuery.firstResult = pageable.offset.toInt()
        val list = listQuery.resultList.map {
            val array = it as Array<*>
            return@map Pair(array[0] as Employee, array[1] as Employer)
        }

        return PageImpl(list, pageable, count)
    }
}