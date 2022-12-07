package com.waynestalk.booksproviderclient

import android.net.Uri

object BooksContract {
    const val WRITE_PERMISSION = "com.waynestalk.booksprovider.provider.READ_BOOKS"
    const val READ_PERMISSION = "com.waynestalk.booksprovider.provider.WRITE_BOOKS"

    val CONTENT_URI: Uri = Uri.parse("content://com.waynestalk.booksprovider.provider/books")

    const val _ID = "id"
    const val NAME = "name"
    const val AUTHORS = "authors"
}