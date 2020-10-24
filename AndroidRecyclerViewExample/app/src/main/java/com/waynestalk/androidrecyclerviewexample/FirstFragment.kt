package com.waynestalk.androidrecyclerviewexample

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.fragment.findNavController
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class FirstFragment : Fragment() {
    private lateinit var viewModel: FirstViewModel

    override fun onCreateView(
            inflater: LayoutInflater,
            container: ViewGroup?,
            savedInstanceState: Bundle?
    ): View? {
        viewModel = ViewModelProvider(this).get(FirstViewModel::class.java)

        val root = inflater.inflate(R.layout.fragment_first, container, false)

        val recyclerView = root.findViewById<RecyclerView>(R.id.recyclerView)
        recyclerView.adapter = FirstAdapter(viewModel.dataset)
        recyclerView.layoutManager = LinearLayoutManager(activity)

        return root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        view.findViewById<Button>(R.id.button_first).setOnClickListener {
            findNavController().navigate(R.id.action_FirstFragment_to_SecondFragment)
        }
    }
}