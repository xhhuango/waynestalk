package com.waynestalk.booksproviderclient

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.waynestalk.booksproviderclient.databinding.AddBookFragmentBinding

class AddBookFragment : Fragment() {
    companion object {
        private const val ARG_BOOK = "book"

        fun newInstance(book: Book? = null) = AddBookFragment().apply {
            arguments = Bundle().apply {
                book?.let { putParcelable(ARG_BOOK, it) }
            }
        }
    }

    private var _binding: AddBookFragmentBinding? = null
    private val binding: AddBookFragmentBinding
        get() = _binding!!

    private val viewModel: AddBookViewModel by viewModels()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = AddBookFragmentBinding.inflate(layoutInflater)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        arguments?.let { bundle ->
            val book: Book = bundle.getParcelable(ARG_BOOK) ?: return@let
            viewModel.editedBookId = book.id
            binding.nameEditText.setText(book.name)
            binding.authorsEditText.setText(book.authors)
        }

        initViews()
        initViewModel()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun initViews() {
        binding.addButton.setOnClickListener {
            context?.let { context ->
                viewModel.saveBook(
                    context.contentResolver,
                    binding.nameEditText.text.toString(),
                    binding.authorsEditText.text?.toString()
                )
            }
        }
    }

    private fun initViewModel() {
        viewModel.result.observe(viewLifecycleOwner) {
            if (it.isSuccess) {
                Toast.makeText(context, "SUCCESS: ${it.getOrNull()}", Toast.LENGTH_LONG).show()
                activity?.onBackPressedDispatcher?.onBackPressed()
            } else {
                Toast.makeText(context, "ERROR: ${it.exceptionOrNull()}", Toast.LENGTH_LONG).show()
            }
        }
    }
}