//
//  Order.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/27.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import Foundation

struct Order {
    let id: UUID
    let name: String
    let price: Double

    var quantity: Int

    func increased() -> Order {
        var copy = self
        copy.quantity += 1
        return copy
    }

    func decreased() -> Order? {
        guard quantity > 1 else {
            return nil
        }

        var copy = self
        copy.quantity -= 1
        return copy
    }
}
