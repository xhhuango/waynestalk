package com.waynestalk.hiltexample

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.waynestalk.hiltexample.databinding.ProductRowBinding
import com.waynestalk.hiltexample.product.Product
import javax.inject.Inject

class ProductListAdapter @Inject constructor() :
    RecyclerView.Adapter<ProductListAdapter.ViewHolder>() {
    inner class ViewHolder(private val binding: ProductRowBinding) :
        RecyclerView.ViewHolder(binding.root) {
        var product: Product? = null
            set(value) {
                field = value
                layout()
            }

        private fun layout() {
            binding.productTextView.text = product?.name
            binding.priceTextView.text = product?.price.toString()
        }
    }

    var list: List<Product> = emptyList()
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ProductRowBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.product = list[position]
    }

    override fun getItemCount(): Int = list.size
}