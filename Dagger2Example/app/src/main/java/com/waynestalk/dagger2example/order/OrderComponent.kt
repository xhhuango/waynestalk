package com.waynestalk.dagger2example.order

import dagger.Component
import javax.inject.Singleton

@Singleton
@Component(modules = [OrderModule::class])
interface OrderComponent {
    fun inject(fragment: OrderFirstFragment)

    fun inject(fragment: OrderSecondFragment)
}