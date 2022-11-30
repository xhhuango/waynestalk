package com.waynestalk.contactexample.rawcontact

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.waynestalk.contactexample.MainActivity
import com.waynestalk.contactexample.databinding.RawContactListFragmentBinding

class RawContactListFragment : Fragment() {
    companion object {
        fun newInstance() = RawContactListFragment()
    }

    private var _binding: RawContactListFragmentBinding? = null
    private val binding: RawContactListFragmentBinding
        get() = _binding!!

    private val viewModel: RawContactListViewModel by viewModels()

    private var adapter: RawContactListAdapter? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = RawContactListFragmentBinding.inflate(layoutInflater)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        adapter = RawContactListAdapter()
        binding.recycleView.adapter = adapter
        binding.recycleView.layoutManager = LinearLayoutManager(context)
        adapter?.onClick = {
            (activity as? MainActivity)?.showContact(it)
        }
        adapter?.onDelete = {
            context?.let { context ->
                viewModel.deleteContact(context.contentResolver, it)
            }
        }

        binding.searchButton.setOnClickListener {
            context?.let { context ->
                viewModel.loadRawContacts(
                    context.contentResolver,
                    binding.searchEditText.text?.toString()
                )
            }
        }

        initViewModel()
    }

    override fun onResume() {
        super.onResume()
        context?.let {
            viewModel.loadRawContacts(it.contentResolver)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun initViewModel() {
        viewModel.rawContacts.observe(viewLifecycleOwner) {
            adapter?.list = it
        }

        viewModel.result.observe(viewLifecycleOwner) {
            if (it.isSuccess) {
                context?.let { context -> viewModel.loadRawContacts(context.contentResolver) }
                Toast.makeText(context, "SUCCESS", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(
                    context,
                    "ERROR: ${it.exceptionOrNull()?.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }
}