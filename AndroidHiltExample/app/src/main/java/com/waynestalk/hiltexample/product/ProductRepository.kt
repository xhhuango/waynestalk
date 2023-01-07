package com.waynestalk.hiltexample.product

interface ProductRepository {
    suspend fun addOrder(product: Product)
    suspend fun getAllOrders(): List<Product>
}