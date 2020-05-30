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
    @State var showingActionSheet = false

    var body: some View {
        Button(action: {
            self.showingActionSheet = true
        }) {
            HStack {
                Text(product.name ?? "")
                    .font(.custom("Georgia", size: 24))
                    .foregroundColor(Color("MenuTextColor"))
                Spacer()
                Text(String(format: "$ %.2f", product.price))
                    .font(.custom("Georgia", size: 24))
                    .foregroundColor(Color("MenuTextColor"))
            }
        }
            .buttonStyle(PlainButtonStyle())
            .actionSheet(isPresented: self.$showingActionSheet) {
                ActionSheet(title: Text(product.name ?? ""), buttons: [
                    .default(Text("Add")) {
                        OrderManager.shared.add(product: self.product)
                    },
                    .destructive(Text("Canel")),
                ])
            }
            .frame(height: 36)
            .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
    }
}

struct MenuRowView_Previews: PreviewProvider {
    private static var product: Product {
        let product = Product()
        product.id = UUID()
        product.name = "Americano"
        product.price = 25
        return product
    }

    static var previews: some View {
        MenuRowView(product: product)
    }
}
