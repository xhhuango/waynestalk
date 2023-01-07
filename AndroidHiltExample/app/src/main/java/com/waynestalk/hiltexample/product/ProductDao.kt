package com.waynestalk.hiltexample.product

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface ProductDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: Product)

    @Query("SELECT * FROM products")
    suspend fun findAll(): List<Product>
}