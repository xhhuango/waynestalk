package com.waynestalk.booksproviderclient

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.waynestalk.booksproviderclient.databinding.BookListFragmentBinding

class BookListFragment : Fragment() {
    companion object {
        fun newInstance() = BookListFragment()
    }

    private var _binding: BookListFragmentBinding? = null
    private val binding: BookListFragmentBinding
        get() = _binding!!

    private val viewModel: BookListViewModel by viewModels()
    private var adapter: BookListAdapter? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = BookListFragmentBinding.inflate(layoutInflater)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        initViews()
        initViewModel()
    }

    override fun onResume() {
        super.onResume()
        context?.let {
            viewModel.loadBooks(
                it.contentResolver,
                binding.searchEditText.text?.toString()
            )
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun initViews() {
        adapter = BookListAdapter()
        binding.recyclerView.adapter = adapter
        binding.recyclerView.layoutManager = LinearLayoutManager(context)
        adapter?.onDelete = {
            context?.let { context -> viewModel.deleteBook(context.contentResolver, it) }
        }
        adapter?.onEdit = { (activity as? MainActivity)?.showAddBook(it) }

        binding.searchButton.setOnClickListener {
            context?.let {
                viewModel.loadBooks(
                    it.contentResolver,
                    binding.searchEditText.text?.toString()
                )
            }
        }

        binding.addButton.setOnClickListener {
            (activity as? MainActivity)?.showAddBook()
        }
    }

    private fun initViewModel() {
        viewModel.books.observe(viewLifecycleOwner) {
            adapter?.list = it
        }

        viewModel.result.observe(viewLifecycleOwner) {
            if (it.isSuccess) {
                Toast.makeText(context, "SUCCESS: ${it.getOrNull()}", Toast.LENGTH_LONG).show()
                context?.let { context -> viewModel.loadBooks(context.contentResolver) }
            } else {
                Toast.makeText(context, "ERROR: ${it.exceptionOrNull()}", Toast.LENGTH_LONG).show()
            }
        }
    }
}