package com.waynestalk.example

import android.content.Context
import androidx.annotation.VisibleForTesting
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import net.sqlcipher.database.SQLiteDatabase
import net.sqlcipher.database.SupportFactory
import java.util.concurrent.Executors

@Database(entities = [Employee::class], version = 1)
@TypeConverters(Converters::class)
abstract class EmployeeDatabase : RoomDatabase() {
    abstract fun dao(): EmployeeDao

    companion object {
        @Volatile
        private var INSTANCE: EmployeeDatabase? = null

        fun getInstance(context: Context, path: String, password: String): EmployeeDatabase {
            return INSTANCE ?: synchronized(this) {
//                val supportFactory = SupportFactory(SQLiteDatabase.getBytes(password.toCharArray()))
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    EmployeeDatabase::class.java,
                    path,
                )
//                    .allowMainThreadQueries()
//                    .openHelperFactory(supportFactory)
                    .setQueryCallback({ sqlQuery, bindArgs ->
                        println("SQL: $sqlQuery; Args: $bindArgs")
                    }, Executors.newSingleThreadExecutor())
                    .build()
                INSTANCE = instance
                instance
            }
        }

        @VisibleForTesting
        @Synchronized
        fun getTestingInstance(context: Context): EmployeeDatabase {
            return Room
                .inMemoryDatabaseBuilder(context.applicationContext, EmployeeDatabase::class.java)
                .build()
        }
    }
}