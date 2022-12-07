package com.waynestalk.booksproviderclient

import android.content.pm.PackageManager
import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.waynestalk.booksproviderclient.databinding.MainActivityBinding

class MainActivity : AppCompatActivity() {
    companion object {
        private const val FRAGMENT_BOOK_LIST = "BookList"
        private const val FRAGMENT_ADD_BOOK = "AddBook"

        private const val PERMISSION_REQUEST_CODE = 1
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

        checkPermissions()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode != PERMISSION_REQUEST_CODE) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults)
            return
        }

        for (index in permissions.indices) {
            if (permissions[index] == BooksContract.WRITE_PERMISSION) {
                if (grantResults[index] == PackageManager.PERMISSION_GRANTED) {
                    return
                }
            }
        }
    }

    private fun checkPermissions() {
        if (isPermissionGranted(BooksContract.READ_PERMISSION) &&
            isPermissionGranted(BooksContract.WRITE_PERMISSION)
        ) {
            return
        }

        requestPermissions()
    }

    private fun isPermissionGranted(permission: String): Boolean =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    private fun requestPermissions() {
        if (shouldShowRequestPermission(BooksContract.WRITE_PERMISSION)) {
            AlertDialog.Builder(this)
                .setMessage("Access to the contacts is required")
                .setPositiveButton(android.R.string.ok) { _, _ -> requestPermission() }
                .setNegativeButton(android.R.string.cancel) { _, _ -> finish() }
                .create()
                .show()
        } else {
            requestPermission()
        }
    }

    private fun shouldShowRequestPermission(permission: String): Boolean =
        ActivityCompat.shouldShowRequestPermissionRationale(this, permission)

    private fun requestPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(BooksContract.READ_PERMISSION, BooksContract.WRITE_PERMISSION),
            PERMISSION_REQUEST_CODE
        )
    }

    fun showAddBook(book: Book? = null) {
        supportFragmentManager.beginTransaction()
            .replace(R.id.container, AddBookFragment.newInstance(book), FRAGMENT_ADD_BOOK)
            .addToBackStack(FRAGMENT_ADD_BOOK)
            .commit()
    }
}