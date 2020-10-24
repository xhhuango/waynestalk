package com.waynestalk.androidrecyclerviewexample

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class FirstAdapter(private val dataset: Array<RowData>) : RecyclerView.Adapter<FirstAdapter.ViewHolder>() {
    inner class ViewHolder(listItemView: View) : RecyclerView.ViewHolder(listItemView) {
        val keyTextView = itemView.findViewById<TextView>(R.id.keyTextView)
        val valueTextView = itemView.findViewById<TextView>(R.id.valueTextView)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        val view = inflater.inflate(R.layout.row, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val rowData = dataset[position]
        holder.keyTextView.text = rowData.key
        holder.valueTextView.text = rowData.value
    }

    override fun getItemCount(): Int {
        return dataset.size
    }
}