//
//  ProductList.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/25.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct ProductListView: View {
    var body: some View {
        VStack {
            Text("Product List")
        }
        .navigationBarTitle("Merchant")
    }
}

struct ProductList_Previews: PreviewProvider {
    static var previews: some View {
        ProductListView()
    }
}
