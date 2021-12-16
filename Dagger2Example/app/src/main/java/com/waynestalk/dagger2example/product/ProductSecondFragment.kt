package com.waynestalk.dagger2example.product

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.waynestalk.dagger2example.R
import com.waynestalk.dagger2example.databinding.FragmentSecondProductBinding
import javax.inject.Inject

class ProductSecondFragment : Fragment() {
    private var _binding: FragmentSecondProductBinding? = null
    private val binding get() = _binding!!

    @Inject
    lateinit var productManager: ProductManager

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSecondProductBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onAttach(context: Context) {
        activity?.let {
            (it as ProductActivity).productComponent.inject(this)
            Log.d(javaClass.canonicalName, "productManager: $productManager")
        }
        super.onAttach(context)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.textviewSecond.text = "productManager: $productManager"
        binding.buttonSecond.setOnClickListener {
            findNavController().navigate(R.id.action_SecondFragment_to_FirstFragment)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}