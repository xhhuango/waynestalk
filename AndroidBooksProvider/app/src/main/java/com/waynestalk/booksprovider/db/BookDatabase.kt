package com.waynestalk.booksprovider.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [BookEntity::class], version = 1)
abstract class BookDatabase : RoomDatabase() {
    abstract fun dao(): BookDao

    companion object {
        private const val FILENAME = "books.db"

        @Volatile
        private var INSTANCE: BookDatabase? = null

        fun getInstance(context: Context): BookDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context,
                    BookDatabase::class.java,
                    context.getDatabasePath(FILENAME).absolutePath,
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}