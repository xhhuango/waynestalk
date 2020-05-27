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
        List {
            ForEach(manager.orders, id: \.id) { order in
                OrderRowView(order: order)
            }
        }
    }
}

struct OrderListView_Previews: PreviewProvider {
    static var previews: some View {
        OrderListView()
    }
}
