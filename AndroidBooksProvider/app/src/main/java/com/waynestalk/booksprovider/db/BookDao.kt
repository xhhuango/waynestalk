package com.waynestalk.booksprovider.db

import android.database.Cursor
import androidx.room.*
import androidx.sqlite.db.SimpleSQLiteQuery

@Dao
interface BookDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insert(entity: BookEntity): Long

    @Update
    fun update(entity: BookEntity): Int

    @RawQuery
    fun query(query: SimpleSQLiteQuery): Cursor
}