package com.waynestalk.hiltexample

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.waynestalk.hiltexample.product.Product
import com.waynestalk.hiltexample.product.ProductRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ProductListViewModel @Inject constructor(private val productRepository: ProductRepository) :
    ViewModel() {
    val orders: MutableLiveData<List<Product>> = MutableLiveData()
    val addResult: MutableLiveData<Result<Unit>> = MutableLiveData()

    fun getAllOrders() {
        viewModelScope.launch(Dispatchers.IO) {
            val list = productRepository.getAllOrders()
            orders.postValue(list)
        }
    }

    fun addProduct(product: Product) {
        viewModelScope.launch(Dispatchers.IO) {
            productRepository.addOrder(product)
            addResult.postValue(Result.success(Unit))
        }
    }
}