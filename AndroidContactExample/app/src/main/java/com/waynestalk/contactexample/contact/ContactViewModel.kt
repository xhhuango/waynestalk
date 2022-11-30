package com.waynestalk.contactexample.contact

import android.content.ContentResolver
import android.content.ContentValues
import android.provider.ContactsContract.CommonDataKinds.Email
import android.provider.ContactsContract.Data
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class ContactViewModel : ViewModel() {
    var rawContactId: Long = 0

    val name: MutableLiveData<String?> = MutableLiveData()
    val emailList: MutableLiveData<List<ContactEmail>> = MutableLiveData()

    val result: MutableLiveData<Result<String>> = MutableLiveData()

    fun loadData(contentResolver: ContentResolver) {
        viewModelScope.launch(Dispatchers.IO) {
            val name = loadName(contentResolver)
            this@ContactViewModel.name.postValue(name)

            val list = loadEmail(contentResolver)
            emailList.postValue(list)
        }
    }

    private fun loadName(contentResolver: ContentResolver): String? {
        val cursor = contentResolver.query(
            Data.CONTENT_URI,
            arrayOf(Data._ID, Data.DISPLAY_NAME_PRIMARY),
            "${Data.RAW_CONTACT_ID} = ?",
            arrayOf("$rawContactId"),
            null,
        ) ?: return null

        if (cursor.count == 0) {
            cursor.close()
            return null
        }

        val nameIndex = cursor.getColumnIndex(Data.DISPLAY_NAME_PRIMARY)

        var name: String? = null
        while (cursor.moveToNext()) {
            name = cursor.getString(nameIndex)
        }

        cursor.close()
        return name
    }

    private fun loadEmail(contentResolver: ContentResolver): List<ContactEmail> {
        val cursor = contentResolver.query(
            Data.CONTENT_URI,
            arrayOf(Email._ID, Email.ADDRESS, Email.TYPE),
            "${Data.RAW_CONTACT_ID} = ? AND ${Data.MIMETYPE} = '${Email.CONTENT_ITEM_TYPE}'",
            arrayOf("$rawContactId"),
            "${Email.TYPE} ASC"
        ) ?: return emptyList()

        if (cursor.count == 0) {
            cursor.close()
            return emptyList()
        }

        val idIndex = cursor.getColumnIndex(Data._ID)
        val emailAddressIndex = cursor.getColumnIndex(Email.ADDRESS)
        val emailTypeIndex = cursor.getColumnIndex(Email.TYPE)

        val list = mutableListOf<ContactEmail>()
        while (cursor.moveToNext()) {
            val id = cursor.getLong(idIndex)
            val emailAddress = cursor.getString(emailAddressIndex)
            val emailType = cursor.getInt(emailTypeIndex)
            list.add(ContactEmail(id, emailAddress, emailType))
        }

        cursor.close()
        return list
    }

    fun addEmail(contentResolver: ContentResolver, email: String, type: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val values = ContentValues()
                values.put(Data.RAW_CONTACT_ID, rawContactId)
                values.put(Data.MIMETYPE, Email.CONTENT_ITEM_TYPE)
                values.put(Email.TYPE, type)
                values.put(Email.ADDRESS, email)
                val uri = contentResolver.insert(Data.CONTENT_URI, values)

                result.postValue(
                    if (uri != null) Result.success(uri.toString())
                    else Result.failure(Exception("Error on inserting email"))
                )
            } catch (e: Exception) {
                result.postValue(Result.failure(e))
            }
        }
    }

    fun saveEmail(contentResolver: ContentResolver, email: String, type: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val contactEmail = emailList.value?.find { it.type == type } ?: return@launch

                val values = ContentValues()
                values.put(Email.ADDRESS, email)
                val count = contentResolver.update(
                    Data.CONTENT_URI,
                    values,
                    "${Data._ID} = ?",
                    arrayOf("${contactEmail.id}")
                )

                result.postValue(
                    if (count == 1) Result.success("Updated email")
                    else Result.failure(Exception("Error on updating email"))
                )
            } catch (e: Exception) {
                result.postValue(Result.failure(e))
            }
        }
    }

    fun removeEmail(contentResolver: ContentResolver, type: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val email = emailList.value?.find { it.type == type } ?: return@launch
                val count = contentResolver.delete(
                    Data.CONTENT_URI,
                    "${Data._ID} = ?",
                    arrayOf("${email.id}")
                )

                result.postValue(
                    if (count == 1) Result.success("Removed email")
                    else Result.failure(Exception("Error on removing email"))
                )
            } catch (e: Exception) {
                result.postValue(Result.failure(e))
            }
        }
    }
}