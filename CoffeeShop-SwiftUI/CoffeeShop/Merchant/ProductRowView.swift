//
//  ProductRowView.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/25.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct ProductRowView: View {
    let product: Product
    
    var body: some View {
        HStack() {
            Text(product.name)
                .font(.custom("Georgia", size: 20))
                .foregroundColor(Color("MenuTextColor"))
            Spacer()
            Text(String(format: "%.2f $", product.price))
                .font(.custom("Georgia", size: 20))
                .foregroundColor(Color("MenuTextColor"))
        }
        .padding(72.0)
        .frame(height: 44.0)
        .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
    }
}

struct ProductRowView_Previews: PreviewProvider {
    static var previews: some View {
        ProductRowView(product: Product(name: "Capuccino", price: 30))
    }
}
