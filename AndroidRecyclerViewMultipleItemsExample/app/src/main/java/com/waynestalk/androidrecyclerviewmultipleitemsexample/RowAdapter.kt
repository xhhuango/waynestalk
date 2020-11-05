package com.waynestalk.androidrecyclerviewmultipleitemsexample

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView

class RowAdapter(private val list: List<RowData>) :
    RecyclerView.Adapter<RecyclerView.ViewHolder>() {
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val rowData = list.find { it.layout == viewType }!!
        val inflater = LayoutInflater.from(parent.context)
        val view = inflater.inflate(viewType, parent, false)
        return rowData.onCreateViewHolder(view)
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val rowData = list[position]
        rowData.onBindViewHolder(holder)
    }

    override fun getItemViewType(position: Int): Int {
        return list[position].layout
    }

    override fun getItemCount(): Int {
        return list.size
    }
}