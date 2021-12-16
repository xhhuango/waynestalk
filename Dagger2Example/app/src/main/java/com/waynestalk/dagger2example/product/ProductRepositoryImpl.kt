package com.waynestalk.dagger2example.product

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ProductRepositoryImpl @Inject constructor() : ProductRepository {
    override fun findAll(): List<Product> = listOf(
        Product("Chocolate", 1.0),
        Product("Jelly Bean", 0.2),
    )
}