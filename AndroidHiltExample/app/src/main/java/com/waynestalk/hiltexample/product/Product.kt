package com.waynestalk.hiltexample.product

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "products")
data class Product(
    val name: String,
    val price: Int,
    @PrimaryKey(autoGenerate = true) var id: Int? = null
)
