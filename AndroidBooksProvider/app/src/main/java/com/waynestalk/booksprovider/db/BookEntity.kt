package com.waynestalk.booksprovider.db

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "books")
data class BookEntity(
    @PrimaryKey val id: Long?,
    val name: String,
    val authors: String?,
)