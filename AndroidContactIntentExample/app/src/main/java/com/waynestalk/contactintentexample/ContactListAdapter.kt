package com.waynestalk.contactintentexample

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.waynestalk.contactintentexample.databinding.ContactRowBinding

class ContactListAdapter : RecyclerView.Adapter<ContactListAdapter.ViewHolder>() {
    var list: List<Account> = emptyList()
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    var onEdit: ((Account) -> Unit)? = null

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ContactRowBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.account = list[position]
    }

    override fun getItemCount(): Int = list.size

    inner class ViewHolder(private val binding: ContactRowBinding) :
        RecyclerView.ViewHolder(binding.root) {
        var account: Account? = null
            set(value) {
                field = value
                layout()
            }

        private fun layout() {
            binding.nameTextView.text = account?.name

            binding.editButton.setOnClickListener {
                account?.let { account -> onEdit?.let { it(account) } }
            }
        }
    }
}