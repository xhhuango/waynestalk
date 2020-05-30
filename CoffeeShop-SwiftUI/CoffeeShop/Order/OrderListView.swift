//
//  OrderListView.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/27.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct OrderListView: View {
    @ObservedObject var manager = OrderManager.shared

    var body: some View {
        VStack {
            List {
                ForEach(manager.orders, id: \.id) { order in
                    OrderRowView(order: order, onIncrement: { order in
                        self.manager.add(order: order)
                    }, onDecrement: { order in
                        self.manager.remove(order: order)
                    })
                }
            }

            HStack {
                Text("Total")
                    .font(.system(size: 22))
                Spacer()
                Text(String(format: "$ %.2f", total()))
                    .font(.custom("Georgia", size: 22))
            }
                .frame(minWidth: 0, maxWidth: .infinity, alignment: .trailing)
                .padding()
        }
    }

    private func total() -> Double {
        manager.orders.reduce(0, { total, order in
            total + order.price * Double(order.quantity)
        })

    }
}

struct OrderListView_Previews: PreviewProvider {
    static var previews: some View {
        OrderListView()
    }
}
