//
//  MenuRowView.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/27.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct MenuRowView: View {
    let product: Product

    var body: some View {
        HStack {
            Text(product.name ?? "")
                .font(.custom("Georgia", size: 24))
                .foregroundColor(Color("MenuTextColor"))
            Spacer()
            Text(String(format: "$ %.2f", product.price))
                .font(.custom("Georgia", size: 24))
                .foregroundColor(Color("MenuTextColor"))
        }
            .frame(height: 36)
            .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
    }
}

struct MenuRowView_Previews: PreviewProvider {
    private static var product: Product {
        let product = Product()
        product.id = UUID()
        product.name = "美式咖啡"
        product.price = 25
        return product
    }

    static var previews: some View {
        MenuRowView(product: product)
    }
}
