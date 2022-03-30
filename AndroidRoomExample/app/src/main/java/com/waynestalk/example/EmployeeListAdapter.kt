package com.waynestalk.example

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.waynestalk.example.databinding.EmployeeRowBinding
import java.text.SimpleDateFormat

class EmployeeListAdapter : RecyclerView.Adapter<EmployeeListAdapter.ViewHolder>() {
    inner class ViewHolder(private val binding: EmployeeRowBinding) :
        RecyclerView.ViewHolder(binding.root) {
        fun bindEmployee(employee: Employee) {
            binding.nameTextView.text = employee.name
            binding.typeTextView.text = employee.type.name
            binding.createdAtTextView.text =
                SimpleDateFormat.getDateInstance().format(employee.createdAt)
        }
    }

    var list: List<Employee> = emptyList()
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = EmployeeRowBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bindEmployee(list[position])
    }

    override fun getItemCount(): Int = list.size
}