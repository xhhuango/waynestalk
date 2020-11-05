package com.waynestalk.androidrecyclerviewmultipleitemsexample

import android.view.View
import android.widget.TextView
import androidx.appcompat.widget.SwitchCompat
import androidx.recyclerview.widget.RecyclerView

class SwitchRowData(
    private val title: String,
    private val value: Boolean,
    private val onCheckedChange: (Boolean) -> Unit
) : RowData {
    inner class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val titleTextView: TextView = itemView.findViewById(R.id.titleTextView)
        val valueSwitch: SwitchCompat = itemView.findViewById(R.id.valueSwitch)
    }

    override val layout = R.layout.row_switch

    override fun onCreateViewHolder(view: View) = ViewHolder(view)

    override fun onBindViewHolder(viewHolder: RecyclerView.ViewHolder) {
        viewHolder as ViewHolder
        viewHolder.titleTextView.text = title
        viewHolder.valueSwitch.isChecked = value
        viewHolder.valueSwitch.setOnCheckedChangeListener { buttonView, isChecked ->
            onCheckedChange(isChecked)
        }
    }
}