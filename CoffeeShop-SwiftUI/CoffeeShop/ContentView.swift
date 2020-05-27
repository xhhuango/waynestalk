//
//  ContentView.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/25.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct ContentView: View {
    @State var selection = 1

    var body: some View {
        TabView(selection: $selection) {
            NavigationView {
                ProductListView()
            }
                .tabItem {
                    Image("ShopIcon")
                    Text("店家")
                }
                .tag(0)


            MenuView()
                .tabItem {
                    Image("MenuIcon")
                    Text("菜單")
                }
                .tag(1)

            OrderListView()
                .tabItem {
                    Image("OrderIcon")
                    Text("點單")
                }
                .tag(2)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
