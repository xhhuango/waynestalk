package com.waynestalk.hiltexample

import com.waynestalk.hiltexample.product.ProductRepository
import com.waynestalk.hiltexample.product.ProductRepositoryImpl
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

@Module
@InstallIn(SingletonComponent::class)
abstract class AppModuleClass {
    @Binds
    abstract fun provideOrderRepository(impl: ProductRepositoryImpl): ProductRepository
}