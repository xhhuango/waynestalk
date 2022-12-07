package com.waynestalk.booksprovider

import android.content.ContentResolver
import android.content.ContentUris
import android.content.ContentValues
import android.net.Uri
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class AddBookViewModel : ViewModel() {
    val result: MutableLiveData<Result<Uri>> = MutableLiveData()

    var editedBookId: Long? = null

    fun saveBook(contentResolver: ContentResolver, name: String, authors: String?) {
        viewModelScope.launch(Dispatchers.IO) {
            if (editedBookId == null) {
                addBook(contentResolver, name, authors)
            } else {
                editBook(contentResolver, name, authors)
            }
        }
    }

    private fun addBook(contentResolver: ContentResolver, name: String, authors: String?) {
        try {
            val values = ContentValues()
            values.put(BooksContract.NAME, name)
            authors?.let { values.put(BooksContract.AUTHORS, it) }

            val uri = contentResolver.insert(BooksContract.CONTENT_URI, values)
            if (uri == null) {
                result.postValue(Result.failure(Exception("Returned URI is null")))
            } else {
                result.postValue(Result.success(uri))
            }
        } catch (e: Exception) {
            result.postValue(Result.failure(e))
        }
    }

    private fun editBook(contentResolver: ContentResolver, name: String, authors: String?) {
        try {
            val values = ContentValues()
            values.put(BooksContract.NAME, name)
            authors?.let { values.put(BooksContract.AUTHORS, it) }

            val uri = ContentUris.withAppendedId(BooksContract.CONTENT_URI, editedBookId!!)
            val rows = contentResolver.update(uri, values, null, null)
            if (rows > 0) {
                result.postValue(Result.success(uri))
            } else {
                result.postValue(Result.failure(Exception("Couldn't update the book")))
            }
        } catch (e: Exception) {
            result.postValue(Result.failure(e))
        }
    }
}