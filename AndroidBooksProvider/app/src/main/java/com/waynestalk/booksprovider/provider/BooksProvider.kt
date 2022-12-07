package com.waynestalk.booksprovider.provider

import android.content.*
import android.database.Cursor
import android.database.sqlite.SQLiteQueryBuilder
import android.net.Uri
import android.util.Log
import androidx.room.OnConflictStrategy
import androidx.sqlite.db.SimpleSQLiteQuery
import com.waynestalk.booksprovider.db.BookDatabase
import com.waynestalk.booksprovider.db.BookEntity

class BooksProvider : ContentProvider() {
    companion object {
        private const val TAG = "BooksProvider"

        private const val AUTHORITY = "com.waynestalk.booksprovider.provider"
        private val CONTENT_URI = Uri.parse("content://${AUTHORITY}/books")

        private const val BOOKS_MIME_TYPE =
            "${ContentResolver.CURSOR_DIR_BASE_TYPE}/vnd.com.waynestalk.booksprovider.provider.books"
        private const val BOOK_MIME_TYPE =
            "${ContentResolver.CURSOR_ITEM_BASE_TYPE}/vnd.com.waynestalk.booksprovider.provider.books"

        private const val FIND_ALL = 1
        private const val FIND_ONE = 2

        private const val _ID = "id"
        private const val NAME = "name"
        private const val AUTHORS = "authors"
    }

    private val uriMatcher = UriMatcher(UriMatcher.NO_MATCH).apply {
        addURI(AUTHORITY, "books", FIND_ALL)
        addURI(AUTHORITY, "books/#", FIND_ONE)
    }

    private lateinit var database: BookDatabase

    override fun onCreate(): Boolean {
        Log.d(TAG, "onCreate: ${Thread.currentThread()}")
        database = BookDatabase.getInstance(context ?: return false)
        return true
    }

    override fun query(
        uri: Uri,
        projection: Array<out String>?,
        selection: String?,
        selectionArgs: Array<out String>?,
        sortOrder: String?
    ): Cursor {
        Log.d(TAG, "onQuery: ${Thread.currentThread()}")

        val query = when (uriMatcher.match(uri)) {
            FIND_ALL -> {
                val builder = SQLiteQueryBuilder()
                builder.tables = "books"
                val sql = builder.buildQuery(
                    projection,
                    selection,
                    null,
                    null,
                    if (sortOrder?.isNotEmpty() == true) sortOrder else "id ASC",
                    null
                )
                SimpleSQLiteQuery(sql, selectionArgs)
            }
            FIND_ONE -> {
                val builder = SQLiteQueryBuilder()
                builder.tables = "books"
                val sql = builder.buildQuery(
                    projection,
                    "id = ${uri.lastPathSegment}",
                    null,
                    null,
                    null,
                    null
                )
                SimpleSQLiteQuery(sql)
            }
            else -> {
                throw IllegalArgumentException("Unsupported URI: $uri")
            }
        }

        val cursor = database.dao().query(query)
        cursor.setNotificationUri(context?.contentResolver, uri)

        return cursor
    }

    override fun getType(uri: Uri): String {
        return when (uriMatcher.match(uri)) {
            FIND_ALL -> BOOKS_MIME_TYPE
            FIND_ONE -> BOOK_MIME_TYPE
            else -> throw IllegalArgumentException("Unsupported URI: $uri")
        }
    }

    override fun insert(uri: Uri, values: ContentValues?): Uri? {
        Log.d(TAG, "insert: ${Thread.currentThread()}")

        if (uriMatcher.match(uri) != FIND_ALL)
            throw IllegalArgumentException("Unsupported URI for insertion: $uri")
        if (values == null) return null

        val name =
            values.getAsString(NAME) ?: throw IllegalArgumentException("Value NAME is required")
        val authors = values.getAsString(AUTHORS)
        val id = database.dao().insert(BookEntity(null, name, authors))
        val entityUri = ContentUris.withAppendedId(CONTENT_URI, id)
        context?.contentResolver?.notifyChange(entityUri, null)
        return entityUri
    }

    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int {
        Log.d(TAG, "delete: ${Thread.currentThread()}")

        return when (uriMatcher.match(uri)) {
            FIND_ALL -> {
                val rows = database.openHelper.writableDatabase.delete(
                    "books",
                    selection,
                    selectionArgs
                )
                context?.contentResolver?.notifyChange(uri, null)
                rows
            }
            FIND_ONE -> {
                val rows = database.openHelper.writableDatabase.delete(
                    "books",
                    "id = ?",
                    arrayOf(uri.lastPathSegment)
                )
                context?.contentResolver?.notifyChange(uri, null)
                rows
            }
            else -> throw IllegalArgumentException("Unsupported URI: $uri")
        }
    }

    override fun update(
        uri: Uri,
        values: ContentValues?,
        selection: String?,
        selectionArgs: Array<out String>?
    ): Int {
        Log.d(TAG, "update: ${Thread.currentThread()}")
        if (values == null) return 0

        return when (uriMatcher.match(uri)) {
            FIND_ALL -> {
                val rows = database.openHelper.writableDatabase.update(
                    "books",
                    OnConflictStrategy.REPLACE,
                    values,
                    selection,
                    selectionArgs
                )
                context?.contentResolver?.notifyChange(uri, null)
                rows
            }
            FIND_ONE -> {
                val name = values.getAsString(NAME)
                val authors = values.getAsString(AUTHORS)
                val id = uri.lastPathSegment?.toLong() ?: return 0
                val rows = database.dao().update(BookEntity(id, name, authors))
                context?.contentResolver?.notifyChange(uri, null)
                rows
            }
            else -> throw IllegalArgumentException("Unsupported URI: $uri")
        }
    }
}