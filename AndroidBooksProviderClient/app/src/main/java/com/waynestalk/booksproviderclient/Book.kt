package com.waynestalk.booksproviderclient

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class Book(
    val id: Long,
    val name: String,
    val authors: String,
) : Parcelable