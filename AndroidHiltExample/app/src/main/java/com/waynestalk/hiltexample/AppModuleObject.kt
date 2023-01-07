package com.waynestalk.hiltexample

import android.content.Context
import androidx.room.Room
import com.waynestalk.hiltexample.product.ProductDao
import com.waynestalk.hiltexample.product.ProductDatabase
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModuleObject {
    @Singleton
    @Provides
    fun provideProductDatabase(@ApplicationContext appContext: Context): ProductDatabase {
        return Room.databaseBuilder(appContext, ProductDatabase::class.java, "products").build()
    }

    @Singleton
    @Provides
    fun provideProductDao(database: ProductDatabase): ProductDao {
        return database.dao()
    }
}