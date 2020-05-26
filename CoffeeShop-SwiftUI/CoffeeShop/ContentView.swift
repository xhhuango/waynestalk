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
                    Text("店家")
                }

            NavigationView {
                MenuView()
            }
                .tabItem {
                    Image("OpenIcon")
                    Text("客人")
                }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
