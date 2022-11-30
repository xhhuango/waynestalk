package com.waynestalk.contactexample.rawcontact

import android.content.ContentResolver
import android.provider.ContactsContract.RawContacts
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class RawContactListViewModel : ViewModel() {
    private val projection = arrayOf(
        RawContacts._ID,
        RawContacts.DISPLAY_NAME_PRIMARY,
    )

    val rawContacts: MutableLiveData<List<RawAccount>> = MutableLiveData()
    val result: MutableLiveData<Result<Boolean>> = MutableLiveData()

    fun loadRawContacts(contentResolver: ContentResolver, pattern: String? = null) {
        viewModelScope.launch(Dispatchers.IO) {
            val cursor = contentResolver.query(
                RawContacts.CONTENT_URI,
                projection,
                pattern?.let { "${RawContacts.DISPLAY_NAME_PRIMARY} LIKE ?" },
                pattern?.let { arrayOf("%$it%") },
                "${RawContacts.ACCOUNT_NAME} ASC",
            ) ?: return@launch

            if (cursor.count == 0) {
                cursor.close()
                rawContacts.postValue(emptyList())
                return@launch
            }

            val idIndex = cursor.getColumnIndex(RawContacts._ID)
            val nameIndex = cursor.getColumnIndex(RawContacts.DISPLAY_NAME_PRIMARY)

            val list = mutableListOf<RawAccount>()
            while (cursor.moveToNext()) {
                val id = cursor.getLong(idIndex)
                val name = cursor.getString(nameIndex)
                list.add(RawAccount(id, name))
            }

            cursor.close()
            rawContacts.postValue(list)
        }
    }

    fun deleteContact(contentResolver: ContentResolver, rawAccount: RawAccount) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val count = contentResolver.delete(
                    RawContacts.CONTENT_URI,
                    "${RawContacts._ID} = ?",
                    arrayOf("${rawAccount.id}")
                )

                result.postValue(
                    if (count == 1) Result.success(true)
                    else Result.failure(Exception("Error on deleting account"))
                )
            } catch (e: Exception) {
                result.postValue(Result.failure(e))
            }
        }
    }
}