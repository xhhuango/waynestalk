//
//  OrderManager.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/27.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import Foundation

final class OrderManager: ObservableObject {
    static let shared = OrderManager()

    @Published var orders: [Order] = []

    func add(product: Product) {
        guard let id = product.id else {
            return
        }

        if let index = orders.firstIndex(where: { $0.id == id }) {
            orders[index] = orders[index].increased()
        } else {
            let order = Order(id: id, name: product.name ?? "", price: product.price, quantity: 1)
            orders.append(order)
        }
    }

    func remove(product: Product) {
        guard let id = product.id, let index = orders.firstIndex(where: { $0.id == id }) else {
            return
        }

        if orders[index].decreased() == nil {
            orders.remove(at: index)
        }
    }
}
