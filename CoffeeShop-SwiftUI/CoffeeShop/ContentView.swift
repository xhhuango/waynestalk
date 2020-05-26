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
                    Text("店家")
                }
            
            NavigationView {
                MenuView()
            }
                .tabItem {
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
