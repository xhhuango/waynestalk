package com.waynestalk.dagger2example.product

import dagger.Component
import javax.inject.Singleton

@Singleton
@Component(modules = [ProductModule::class])
interface ProductComponent {
    fun inject(fragment: ProductFirstFragment)

    fun inject(fragment: ProductSecondFragment)
}