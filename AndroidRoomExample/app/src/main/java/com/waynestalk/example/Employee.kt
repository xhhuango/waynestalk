package com.waynestalk.example

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.*

@Entity(tableName = "employees")
data class Employee(
    val name: String,
    val type: Type,
    @ColumnInfo(name = "created_at") val createdAt: Date = Date(),
    @PrimaryKey(autoGenerate = true) var id: Int = 0,
) {
    enum class Type {
        FULL_TIME, PART_TIME,
    }
}
