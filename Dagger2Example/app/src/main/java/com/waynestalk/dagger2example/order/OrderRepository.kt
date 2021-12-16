package com.waynestalk.dagger2example.order

interface OrderRepository {
    fun findAll(): List<Order>
}