package com.waynestalk.dagger2example.order

class OrderManagerImpl(private val repository: OrderRepository) : OrderManager {
    override fun getList(): List<Order> = repository.findAll()
}