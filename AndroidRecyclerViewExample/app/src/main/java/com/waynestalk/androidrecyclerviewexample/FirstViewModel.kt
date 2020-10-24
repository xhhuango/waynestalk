package com.waynestalk.androidrecyclerviewexample

import androidx.lifecycle.ViewModel

class FirstViewModel : ViewModel() {
    val dataset = arrayOf(
            RowData("Side name", "Wayne's Talk"),
            RowData("URL", "https://waynestalk.com"),
    )
}