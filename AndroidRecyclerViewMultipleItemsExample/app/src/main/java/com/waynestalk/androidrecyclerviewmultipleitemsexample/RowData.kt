package com.waynestalk.androidrecyclerviewmultipleitemsexample

import android.view.View
import androidx.recyclerview.widget.RecyclerView

interface RowData {
    val layout: Int
    fun onCreateViewHolder(view: View): RecyclerView.ViewHolder
    fun onBindViewHolder(viewHolder: RecyclerView.ViewHolder)
}