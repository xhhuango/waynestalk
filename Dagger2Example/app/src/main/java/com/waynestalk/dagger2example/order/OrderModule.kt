package com.waynestalk.dagger2example.order

import dagger.Module
import dagger.Provides
import javax.inject.Singleton

@Module
class OrderModule {
    @Singleton
    @Provides
    fun orderRepository(): OrderRepository = OrderRepositoryImpl()

    @Singleton
    @Provides
    fun orderManager(repository: OrderRepository): OrderManager = OrderManagerImpl(repository)
}