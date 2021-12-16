package com.waynestalk.dagger2example.product

import dagger.Binds
import dagger.Module

@Module
abstract class ProductModule {
    @Binds
    abstract fun productRepository(productRepository: ProductRepositoryImpl): ProductRepository

    @Binds
    abstract fun productManager(productManager: ProductManagerImpl): ProductManager
}