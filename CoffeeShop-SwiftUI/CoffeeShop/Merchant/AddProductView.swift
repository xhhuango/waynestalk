//
//  AddProductView.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/26.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct AddProductView: View {
    @State var name: String = ""
    @State var price: String = ""

    var body: some View {
        VStack {
            HStack {
                Text("品名：")
                TextField("輸入商品名稱", text: $name)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
            }
                .padding()

            HStack {
                Text("價格：")
                TextField("輸入商品價格", text: $price)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .keyboardType(.decimalPad)
            }
                .padding()

            Button(action: {

            }, label: { Text("新增") })
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color("ButtonColor"))
                .foregroundColor(Color.white)
                .cornerRadius(30)
                .font(.headline)
                .padding()

            Spacer()
        }
            .navigationBarTitle("新增商品")
    }
}

struct AddProductView_Previews: PreviewProvider {
    static var previews: some View {
        AddProductView()
    }
}
