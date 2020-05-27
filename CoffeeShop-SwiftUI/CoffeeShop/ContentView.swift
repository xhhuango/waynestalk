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

            NavigationView {
                GeometryReader { proxy in
                    MenuView()
                        .frame(width: proxy.size.width, height: proxy.size.height, alignment: .topLeading)
                }
                    .navigationBarTitle("")
                    .navigationBarHidden(true)
            }
                .tabItem {
                    Image("OpenIcon")
                    Text("客人")
                }
                .tag(1)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
