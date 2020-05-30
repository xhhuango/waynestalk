//
//  ContentView.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/25.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            NavigationView {
                ProductListView()
            }
                .tabItem {
                    Image("ShopIcon")
                    Text("Shop")
                }

            MenuView()
                .tabItem {
                    Image("MenuIcon")
                    Text("Menu")
                }

            OrderListView()
                .tabItem {
                    Image("OrderIcon")
                    Text("Bill")
                }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
