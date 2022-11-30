package com.waynestalk.contactexample

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.waynestalk.contactexample.contact.ContactFragment
import com.waynestalk.contactexample.databinding.MainActivityBinding
import com.waynestalk.contactexample.rawcontact.RawAccount
import com.waynestalk.contactexample.rawcontact.RawContactListFragment

class MainActivity : AppCompatActivity() {
    companion object {
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
                .replace(R.id.container, RawContactListFragment.newInstance())
                .commitNow()
        }
    }

    override fun onResume() {
        super.onResume()
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
            if (permissions[index] == Manifest.permission.WRITE_CONTACTS) {
                if (grantResults[index] == PackageManager.PERMISSION_GRANTED) {
                    return
                }
            }
        }

        finish()
    }

    private fun checkPermissions() {
        if (isPermissionGranted(Manifest.permission.READ_CONTACTS) &&
            isPermissionGranted(Manifest.permission.WRITE_CONTACTS)
        ) {
            return
        }

        requestPermissions()
    }

    private fun isPermissionGranted(permission: String): Boolean =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    private fun requestPermissions() {
        if (shouldShowRequestPermission(Manifest.permission.WRITE_CONTACTS)) {
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
            arrayOf(Manifest.permission.READ_CONTACTS, Manifest.permission.WRITE_CONTACTS),
            PERMISSION_REQUEST_CODE
        )
    }

    fun showContact(rawAccount: RawAccount) {
        supportFragmentManager.beginTransaction()
            .replace(R.id.container, ContactFragment.newInstance(rawAccount.id))
            .commitNow()
    }
}