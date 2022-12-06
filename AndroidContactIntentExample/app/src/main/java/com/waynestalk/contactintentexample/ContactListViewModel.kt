package com.waynestalk.contactintentexample

import android.content.ContentResolver
import android.content.Intent
import android.provider.ContactsContract
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class ContactListViewModel : ViewModel() {
    val accounts: MutableLiveData<List<Account>> = MutableLiveData()

    fun loadContacts(contentResolver: ContentResolver) {
        viewModelScope.launch(Dispatchers.IO) {
            val cursor = contentResolver.query(
                ContactsContract.Contacts.CONTENT_URI,
                arrayOf(
                    ContactsContract.Contacts._ID,
                    ContactsContract.Contacts.LOOKUP_KEY,
                    ContactsContract.Contacts.DISPLAY_NAME_PRIMARY
                ),
                null,
                null,
                "${ContactsContract.Contacts.DISPLAY_NAME_PRIMARY} ASC",
            )

            if (cursor == null) {
                accounts.postValue(emptyList())
                return@launch
            } else if (cursor.count == 0) {
                cursor.close()
                accounts.postValue(emptyList())
                return@launch
            }

            val idIndex = cursor.getColumnIndex(ContactsContract.Contacts._ID)
            val lookupKeyIndex = cursor.getColumnIndex(ContactsContract.Contacts.LOOKUP_KEY)
            val nameIndex = cursor.getColumnIndex(ContactsContract.Contacts.DISPLAY_NAME_PRIMARY)

            val list = mutableListOf<Account>()
            while (cursor.moveToNext()) {
                val id = cursor.getLong(idIndex)
                val lookupKey = cursor.getString(lookupKeyIndex)
                val name = cursor.getString(nameIndex)
                list.add(Account(id, lookupKey, name))
            }

            cursor.close()
            accounts.postValue(list)
        }
    }

    fun addContact(): Intent {
        return Intent(ContactsContract.Intents.Insert.ACTION).apply {
            type = ContactsContract.RawContacts.CONTENT_TYPE

            // Sets the special extended data for navigation
            putExtra("finishActivityOnSaveCompleted", true)

            // Insert an email address
//            putExtra(ContactsContract.Intents.Insert.EMAIL, "waynestalk@gmail.com")
//            putExtra(
//                ContactsContract.Intents.Insert.EMAIL_TYPE,
//                ContactsContract.CommonDataKinds.Email.TYPE_WORK
//            )

            // Insert a phone number
//            putExtra(ContactsContract.Intents.Insert.PHONE, "123456789")
//            putExtra(
//                ContactsContract.Intents.Insert.PHONE_TYPE,
//                ContactsContract.CommonDataKinds.Phone.TYPE_WORK
//            )
        }
    }

    fun editContact(account: Account): Intent {
        return Intent(Intent.ACTION_EDIT).apply {
            val contactUri = ContactsContract.Contacts.getLookupUri(account.id, account.lookupKey)
            setDataAndType(contactUri, ContactsContract.Contacts.CONTENT_ITEM_TYPE)

            // Sets the special extended data for navigation
            putExtra("finishActivityOnSaveCompleted", true)

            // Insert an email address
//            putExtra(ContactsContract.Intents.Insert.EMAIL, "waynestalk@gmail.com")
//            putExtra(
//                ContactsContract.Intents.Insert.EMAIL_TYPE,
//                ContactsContract.CommonDataKinds.Email.TYPE_WORK
//            )

            // Insert a phone number
//            putExtra(ContactsContract.Intents.Insert.PHONE, "123456789")
//            putExtra(
//                ContactsContract.Intents.Insert.PHONE_TYPE,
//                ContactsContract.CommonDataKinds.Phone.TYPE_WORK
//            )
        }
    }
}