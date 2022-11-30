package com.waynestalk.contactexample.rawcontact

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.waynestalk.contactexample.databinding.RawContactListRawBinding

class RawContactListAdapter : RecyclerView.Adapter<RawContactListAdapter.ViewHolder>() {
    var list: List<RawAccount> = emptyList()
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    var onClick: ((RawAccount) -> Unit)? = null
    var onDelete: ((RawAccount) -> Unit)? = null

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding =
            RawContactListRawBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.rawAccount = list[position]
    }

    override fun getItemCount(): Int = list.size

    inner class ViewHolder(private val binding: RawContactListRawBinding) :
        RecyclerView.ViewHolder(binding.root) {
        var rawAccount: RawAccount? = null
            set(value) {
                field = value
                layout()
            }

        private fun layout() {
            binding.root.setOnClickListener {
                rawAccount?.let { account -> onClick?.let { it(account) } }
            }

            binding.nameTextView.text = rawAccount?.name

            binding.nameButton.setOnClickListener {
                rawAccount?.let { account -> onDelete?.let { it(account) } }
            }
        }
    }
}