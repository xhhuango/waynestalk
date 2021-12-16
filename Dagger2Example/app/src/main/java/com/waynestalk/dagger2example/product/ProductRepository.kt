package com.waynestalk.dagger2example.product

interface ProductRepository {
    fun findAll(): List<Product>
}