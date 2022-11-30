package com.waynestalk.contactexample.contact

import android.os.Bundle
import android.provider.ContactsContract.CommonDataKinds.Email
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.widget.AppCompatButton
import androidx.appcompat.widget.AppCompatEditText
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.waynestalk.contactexample.databinding.ContactFragmentBinding

class ContactFragment : Fragment() {
    companion object {
        private const val ARG_RAW_CONTACT_ID = "rawContactId"

        fun newInstance(rawContactId: Long) = ContactFragment().apply {
            arguments = Bundle().apply { putLong(ARG_RAW_CONTACT_ID, rawContactId) }
        }
    }

    private var _binding: ContactFragmentBinding? = null
    private val binding: ContactFragmentBinding
        get() = _binding!!

    private val viewModel: ContactViewModel by viewModels()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = ContactFragmentBinding.inflate(layoutInflater)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        arguments?.let {
            viewModel.rawContactId = it.getLong(ARG_RAW_CONTACT_ID, 0)
        }

        initViews()
        initViewModel()
    }

    override fun onResume() {
        super.onResume()
        context?.let {
            viewModel.loadData(it.contentResolver)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun initViews() {
        binding.editHomeEmailButton.setOnClickListener {
            val email = binding.homeEmailEditText.text?.toString() ?: return@setOnClickListener
            context?.let { viewModel.saveEmail(it.contentResolver, email, Email.TYPE_HOME) }
        }
        binding.removeHomeEmailButton.setOnClickListener {
            if (binding.editHomeEmailButton.isEnabled) {
                context?.let { viewModel.removeEmail(it.contentResolver, Email.TYPE_HOME) }
            } else {
                val email = binding.homeEmailEditText.text?.toString() ?: return@setOnClickListener
                context?.let { viewModel.addEmail(it.contentResolver, email, Email.TYPE_HOME) }
            }
        }

        binding.editWorkEmailButton.setOnClickListener {
            val email = binding.workEmailEditText.text?.toString() ?: return@setOnClickListener
            context?.let { viewModel.saveEmail(it.contentResolver, email, Email.TYPE_WORK) }
        }
        binding.removeWorkEmailButton.setOnClickListener {
            if (binding.editWorkEmailButton.isEnabled) {
                context?.let { viewModel.removeEmail(it.contentResolver, Email.TYPE_WORK) }
            } else {
                val email = binding.workEmailEditText.text?.toString() ?: return@setOnClickListener
                context?.let { viewModel.addEmail(it.contentResolver, email, Email.TYPE_WORK) }
            }
        }

        binding.editMobileEmailButton.setOnClickListener {
            val email = binding.mobileEmailEditText.text?.toString() ?: return@setOnClickListener
            context?.let { viewModel.saveEmail(it.contentResolver, email, Email.TYPE_MOBILE) }
        }
        binding.removeMobileEmailButton.setOnClickListener {
            if (binding.editMobileEmailButton.isEnabled) {
                context?.let { viewModel.removeEmail(it.contentResolver, Email.TYPE_MOBILE) }
            } else {
                val email =
                    binding.mobileEmailEditText.text?.toString() ?: return@setOnClickListener
                context?.let { viewModel.addEmail(it.contentResolver, email, Email.TYPE_MOBILE) }
            }
        }

        binding.editOtherEmailButton.setOnClickListener {
            val email = binding.otherEmailEditText.text?.toString() ?: return@setOnClickListener
            context?.let { viewModel.saveEmail(it.contentResolver, email, Email.TYPE_OTHER) }
        }
        binding.removeOtherEmailButton.setOnClickListener {
            if (binding.editOtherEmailButton.isEnabled) {
                context?.let { viewModel.removeEmail(it.contentResolver, Email.TYPE_OTHER) }
            } else {
                val email = binding.otherEmailEditText.text?.toString() ?: return@setOnClickListener
                context?.let { viewModel.addEmail(it.contentResolver, email, Email.TYPE_OTHER) }
            }
        }
    }

    private fun initViewModel() {
        viewModel.name.observe(viewLifecycleOwner) { name ->
            binding.nameTextView.text = name
        }

        viewModel.result.observe(viewLifecycleOwner) {
            if (it.isSuccess) {
                context?.let { context -> viewModel.loadData(context.contentResolver) }
                Toast.makeText(context, "SUCCESS: ${it.getOrNull()}", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(
                    context,
                    "ERROR: ${it.exceptionOrNull()?.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

        viewModel.emailList.observe(viewLifecycleOwner) { list ->
            show(
                list.find { it.type == Email.TYPE_HOME }?.email,
                binding.homeEmailEditText,
                binding.editHomeEmailButton,
                binding.removeHomeEmailButton
            )
            show(
                list.find { it.type == Email.TYPE_WORK }?.email,
                binding.workEmailEditText,
                binding.editWorkEmailButton,
                binding.removeWorkEmailButton
            )
            show(
                list.find { it.type == Email.TYPE_MOBILE }?.email,
                binding.mobileEmailEditText,
                binding.editMobileEmailButton,
                binding.removeMobileEmailButton
            )
            show(
                list.find { it.type == Email.TYPE_OTHER }?.email,
                binding.otherEmailEditText,
                binding.editOtherEmailButton,
                binding.removeOtherEmailButton
            )
        }
    }

    private fun show(
        text: String?,
        editText: AppCompatEditText,
        editButton: AppCompatButton,
        removeButton: AppCompatButton
    ) {
        if (text != null) {
            editText.setText(text)
            editButton.isEnabled = true
            removeButton.isEnabled = true
            removeButton.text = "Remove"
        } else {
            editText.setText("")
            editButton.isEnabled = false
            removeButton.isEnabled = true
            removeButton.text = "Add"
        }
    }
}