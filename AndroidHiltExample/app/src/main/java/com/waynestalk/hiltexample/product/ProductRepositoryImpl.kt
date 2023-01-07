package com.waynestalk.hiltexample.product

import javax.inject.Inject

class ProductRepositoryImpl @Inject constructor(private val dao: ProductDao) : ProductRepository {
    override suspend fun addOrder(product: Product) {
        dao.insert(product)
    }

    override suspend fun getAllOrders(): List<Product> {
        return dao.findAll()
    }
}