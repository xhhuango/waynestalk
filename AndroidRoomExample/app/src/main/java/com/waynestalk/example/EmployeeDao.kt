package com.waynestalk.example

import androidx.room.*
import androidx.sqlite.db.SimpleSQLiteQuery

@Dao
abstract class EmployeeDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    abstract suspend fun insert(employee: Employee)

    @Update
    abstract suspend fun update(employee: Employee)

    @Delete
    abstract suspend fun delete(employee: Employee)

    @Query("SELECT * FROM employees")
    abstract suspend fun findAll(): List<Employee>

    @Query("SELECT * FROM employees WHERE name = :name")
    abstract suspend fun findByName(name: String): List<Employee>

    @RawQuery
    abstract suspend fun execSelect(query: SimpleSQLiteQuery): List<Employee>

    suspend fun findByNameOptional(name: String?): List<Employee> {
        var sql = "SELECT * FROM employees"
        name?.let {
            sql += " WHERE name = $it"
        }

        val query = SimpleSQLiteQuery(sql)
        return execSelect(query)
    }

    @Transaction
    suspend fun delete(list: List<Employee>) {
        list.forEach { delete(it) }
    }
}