//
//  Product.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/25.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import Foundation

struct Product: Identifiable {
    let id = UUID()
    let name: String
    let price: Double
}
