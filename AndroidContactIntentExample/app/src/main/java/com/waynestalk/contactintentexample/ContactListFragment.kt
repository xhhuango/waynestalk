package com.waynestalk.contactintentexample

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.waynestalk.contactintentexample.databinding.ContactListFragmentBinding

class ContactListFragment : Fragment() {
    companion object {
        fun newInstance() = ContactListFragment()
    }

    private lateinit var binding: ContactListFragmentBinding

    private val viewModel: ContactListViewModel by viewModels()
    private var adapter: ContactListAdapter? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        binding = ContactListFragmentBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        initViews()
        initViewModel()
    }

    override fun onResume() {
        super.onResume()
        context?.let { viewModel.loadContacts(it.contentResolver) }
    }

    private fun initViews() {
        adapter = ContactListAdapter()
        binding.recycleView.adapter = adapter
        binding.recycleView.layoutManager = LinearLayoutManager(context)
        adapter?.onEdit = {
            val intent = viewModel.editContact(it)
            startActivity(intent)
        }

        binding.addButton.setOnClickListener {
            val intent = viewModel.addContact()
            startActivity(intent)
        }
    }

    private fun initViewModel() {
        viewModel.accounts.observe(viewLifecycleOwner) {
            adapter?.list = it
        }
    }
}