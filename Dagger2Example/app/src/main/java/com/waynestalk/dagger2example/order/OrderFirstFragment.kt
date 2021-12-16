package com.waynestalk.dagger2example.order

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.waynestalk.dagger2example.R
import com.waynestalk.dagger2example.databinding.FragmentFirstOrderBinding
import javax.inject.Inject

class OrderFirstFragment : Fragment() {
    private var _binding: FragmentFirstOrderBinding? = null
    private val binding get() = _binding!!

    @Inject
    lateinit var orderManager: OrderManager

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentFirstOrderBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onAttach(context: Context) {
        activity?.let {
            (it as OrderActivity).orderComponent.inject(this)
            Log.d(javaClass.canonicalName, "orderManager: $orderManager")
        }
        super.onAttach(context)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.textviewFirst.text = "orderManager: $orderManager"
        binding.buttonFirst.setOnClickListener {
            findNavController().navigate(R.id.action_FirstFragment_to_SecondFragment)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}