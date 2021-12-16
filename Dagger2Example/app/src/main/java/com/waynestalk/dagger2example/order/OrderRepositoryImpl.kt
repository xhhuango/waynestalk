package com.waynestalk.dagger2example.order

class OrderRepositoryImpl : OrderRepository {
    override fun findAll(): List<Order> = listOf(
        Order("CocoCola", 1.5),
        Order("Fries", 1.25),
        Order("Burger", 5.59),
    )
}