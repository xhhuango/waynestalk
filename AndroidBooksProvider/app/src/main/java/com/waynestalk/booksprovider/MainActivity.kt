package com.waynestalk.booksprovider

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.waynestalk.booksprovider.databinding.MainActivityBinding

class MainActivity : AppCompatActivity() {
    companion object {
        private const val FRAGMENT_BOOK_LIST = "BookList"
        private const val FRAGMENT_ADD_BOOK = "AddBook"
    }

    private var _binding: MainActivityBinding? = null
    private val binding: MainActivityBinding
        get() = _binding!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        _binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.container, BookListFragment.newInstance(), FRAGMENT_BOOK_LIST)
                .addToBackStack(FRAGMENT_BOOK_LIST)
                .commit()
        }
    }

    fun showAddBook(book: Book? = null) {
        supportFragmentManager.beginTransaction()
            .replace(R.id.container, AddBookFragment.newInstance(book), FRAGMENT_ADD_BOOK)
            .addToBackStack(FRAGMENT_ADD_BOOK)
            .commit()
    }
}