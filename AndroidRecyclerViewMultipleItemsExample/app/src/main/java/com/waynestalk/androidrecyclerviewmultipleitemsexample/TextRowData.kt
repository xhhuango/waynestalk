package com.waynestalk.androidrecyclerviewmultipleitemsexample

import android.view.View
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class TextRowData(private val title: String, private val value: String) : RowData {
    inner class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val titleTextView: TextView = itemView.findViewById(R.id.titleTextView)
        val valueTextView: TextView = itemView.findViewById(R.id.valueTextView)
    }

    override val layout = R.layout.row_text

    override fun onCreateViewHolder(view: View) = ViewHolder(view)

    override fun onBindViewHolder(viewHolder: RecyclerView.ViewHolder) {
        viewHolder as ViewHolder
        viewHolder.titleTextView.text = title
        viewHolder.valueTextView.text = value
    }
}