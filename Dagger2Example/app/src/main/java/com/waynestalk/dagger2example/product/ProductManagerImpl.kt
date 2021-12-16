package com.waynestalk.dagger2example.product

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ProductManagerImpl @Inject constructor(private val repository: ProductRepository) :
    ProductManager {
    override fun getAll(): List<Product> = repository.findAll()
}