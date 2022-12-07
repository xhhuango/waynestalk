package com.waynestalk.booksproviderclient

import android.content.ContentResolver
import android.content.ContentUris
import android.net.Uri
import android.util.Log
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class BookListViewModel : ViewModel() {
    companion object {
        private const val TAG = "BookListViewModel"
    }

    val books: MutableLiveData<List<Book>> = MutableLiveData()
    val result: MutableLiveData<Result<Uri>> = MutableLiveData()

    fun loadBooks(contentResolver: ContentResolver, namePattern: String? = null) {
        viewModelScope.launch(Dispatchers.IO) {
            val cursor = contentResolver.query(
                BooksContract.CONTENT_URI,
                arrayOf(BooksContract._ID, BooksContract.NAME, BooksContract.AUTHORS),
                namePattern?.takeIf { it.isNotEmpty() }?.let { "${BooksContract.NAME} LIKE ?" },
                namePattern?.takeIf { it.isNotEmpty() }?.let { arrayOf("%$it%") },
                "${BooksContract._ID} ASC",
            ) ?: return@launch

            if (cursor.count == 0) {
                cursor.close()
                books.postValue(emptyList())
                return@launch
            }

            val idIndex = cursor.getColumnIndex(BooksContract._ID)
            val nameIndex = cursor.getColumnIndex(BooksContract.NAME)
            val authorsIndex = cursor.getColumnIndex(BooksContract.AUTHORS)

            val list = mutableListOf<Book>()
            while (cursor.moveToNext()) {
                val id = cursor.getLong(idIndex)
                val name = cursor.getString(nameIndex)
                val authors = cursor.getString(authorsIndex)
                list.add(Book(id, name, authors))
            }
            Log.d(TAG, "Loaded books: $list")

            cursor.close()
            books.postValue(list)
        }
    }

    fun deleteBook(contentResolver: ContentResolver, book: Book) {
        viewModelScope.launch {
            try {
                val uri = ContentUris.withAppendedId(BooksContract.CONTENT_URI, book.id)
                val rows = contentResolver.delete(uri, null, null)
                if (rows > 0) {
                    result.postValue(Result.success(uri))
                } else {
                    result.postValue(Result.failure(Exception("Couldn't delete the book")))
                }
            } catch (e: Exception) {
                result.postValue(Result.failure(e))
            }
        }
    }
}