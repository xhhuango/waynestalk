//
//  OrderRowView.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/27.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct OrderRowView: View {
    let order: Order

    var body: some View {
        HStack {
            Text(order.name)
                .font(.custom("Georgia", size: 20))
            Spacer()
            Text(String(format: "$ %.2f", order.price))
                .font(.custom("Georgia", size: 20))
            Text(String(format: "%d", order.quantity))
        }
            .frame(height: 44)
            .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
    }
}

struct OrderRowView_Previews: PreviewProvider {
    private static var order: Order {
        Order(id: UUID(), name: "美式咖啡", price: 25, quantity: 1)
    }

    static var previews: some View {
        OrderRowView(order: order)
    }
}
