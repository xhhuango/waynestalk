package com.waynestalk.booksprovider

import android.net.Uri

object BooksContract {
    val CONTENT_URI: Uri = Uri.parse("content://com.waynestalk.booksprovider.provider/books")

    const val _ID = "id"
    const val NAME = "name"
    const val AUTHORS = "authors"
}