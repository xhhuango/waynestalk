package com.waynestalk.example

import android.os.Bundle
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.waynestalk.example.databinding.EmployeeListActivityBinding
import kotlinx.coroutines.launch

class EmployeeListActivity : AppCompatActivity() {
    private lateinit var binding: EmployeeListActivityBinding
    private lateinit var viewModel: EmployeeListViewModel
    private lateinit var adapter: EmployeeListAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = EmployeeListActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initViews()
        initRecycleView()
        initViewModel()
    }

    private fun initViews() {
        val spinnerItems = Employee.Type.values().map { it.name }
        binding.typeSpinner.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, spinnerItems)

        binding.addButton.setOnClickListener {
            val name = binding.nameEditText.text?.toString() ?: return@setOnClickListener
            if (name.isEmpty()) return@setOnClickListener
            val type = Employee.Type.valueOf(binding.typeSpinner.selectedItem as String)
            lifecycleScope.launch {
                viewModel.addEmployee(name, type)
            }
        }

        binding.searchByNameButton.setOnClickListener {
            val name = binding.searchByNameEditText.text?.toString() ?: return@setOnClickListener
            lifecycleScope.launch {
                viewModel.searchByName(name)
            }
        }
    }

    private fun initRecycleView() {
        adapter = EmployeeListAdapter()
        binding.recycleView.adapter = adapter
        binding.recycleView.layoutManager =
            LinearLayoutManager(this, LinearLayoutManager.VERTICAL, false)
    }

    private fun initViewModel() {
        viewModel = ViewModelProvider(this)[EmployeeListViewModel::class.java]

        viewModel.initDatabase(this)

        viewModel.employees.observe(this) {
            adapter.list = it
        }
    }
}