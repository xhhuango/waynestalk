//
//  ProductList.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/25.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct ProductListView: View {
    @FetchRequest(entity: Product.entity(), sortDescriptors: [NSSortDescriptor(keyPath: \Product.name, ascending: false)])
    var products: FetchedResults<Product>

    var body: some View {
        List(products, id: \Product.id) { product in
            ProductRowView(product: product)
        }
            .navigationBarTitle("商品列表")
            .navigationBarItems(trailing: NavigationLink(destination: AddProductView()) {
                Image(systemName: "plus")
            })
    }
}

struct ProductList_Previews: PreviewProvider {
    static var previews: some View {
        ProductListView()
    }
}
