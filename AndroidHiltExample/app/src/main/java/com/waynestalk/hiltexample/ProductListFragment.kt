package com.waynestalk.hiltexample

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.waynestalk.hiltexample.databinding.ProductListFragmentBinding
import com.waynestalk.hiltexample.product.Product
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class ProductListFragment : Fragment() {
    companion object {
        fun newInstance() = ProductListFragment()
    }

    private var _binding: ProductListFragmentBinding? = null
    private val binding: ProductListFragmentBinding
        get() = _binding!!

    @Inject
    lateinit var adapter: ProductListAdapter

    private val viewModel: ProductListViewModel by viewModels()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = ProductListFragmentBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.recycleView.adapter = adapter
        binding.recycleView.layoutManager =
            LinearLayoutManager(context, LinearLayoutManager.VERTICAL, false)

        binding.addButton.setOnClickListener {
            val name = binding.productNameEditText.text?.toString() ?: return@setOnClickListener
            val price = binding.priceEditText.text?.toString() ?: return@setOnClickListener
            viewModel.addProduct(Product(name, price.toInt()))
        }

        viewModel.orders.observe(viewLifecycleOwner) {
            adapter.list = it ?: emptyList()
        }

        viewModel.addResult.observe(viewLifecycleOwner) {
            if (it.isSuccess) {
                viewModel.getAllOrders()
            }
        }
    }

    override fun onResume() {
        super.onResume()

        viewModel.getAllOrders()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}