package com.waynestalk.booksprovider

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.waynestalk.booksprovider.databinding.BookListRowBinding

class BookListAdapter : RecyclerView.Adapter<BookListAdapter.ViewHolder>() {
    var list: List<Book> = emptyList()
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    var onDelete: ((Book) -> Unit)? = null
    var onEdit: ((Book) -> Unit)? = null

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = BookListRowBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.book = list[position]
    }

    override fun getItemCount(): Int = list.size

    inner class ViewHolder(private val binding: BookListRowBinding) :
        RecyclerView.ViewHolder(binding.root) {
        var book: Book? = null
            set(value) {
                field = value
                layout()
            }

        private fun layout() {
            binding.nameTextView.text = book?.name
            binding.authorsTextView.text = book?.authors

            binding.deleteButton.setOnClickListener {
                book?.let { book -> onDelete?.let { it(book) } }
            }

            binding.editButton.setOnClickListener {
                book?.let { book -> onEdit?.let { it(book) } }
            }
        }
    }
}