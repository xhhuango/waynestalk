package com.waynestalk.androidrecyclerviewmultipleitemsexample

import android.text.Editable
import android.view.View
import android.widget.EditText
import android.widget.TextView
import androidx.core.widget.doAfterTextChanged
import androidx.recyclerview.widget.RecyclerView

class EditRowData(
    private val title: String,
    private val value: String,
    private val doAfterTextChanged: (text: String) -> Unit
) : RowData {
    inner class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val titleTextView: TextView = itemView.findViewById(R.id.titleTextView)
        val valueEditText: EditText = itemView.findViewById(R.id.valueEditText)
    }

    override val layout = R.layout.row_edit

    override fun onCreateViewHolder(view: View) = ViewHolder(view)

    override fun onBindViewHolder(viewHolder: RecyclerView.ViewHolder) {
        viewHolder as ViewHolder
        viewHolder.titleTextView.text = title
        viewHolder.valueEditText.text = Editable.Factory.getInstance().newEditable(value)
        viewHolder.valueEditText.doAfterTextChanged { it.toString().let(doAfterTextChanged) }
    }
}