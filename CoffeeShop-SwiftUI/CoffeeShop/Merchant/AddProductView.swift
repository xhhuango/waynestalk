//
//  AddProductView.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/26.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct AddProductView: View {
    @Environment(\.managedObjectContext) var context
    @Environment(\.presentationMode) var presentationMode

    @State var name: String = ""
    @State var price: String = ""

    @State var isFailed = false

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
                let isSuccessful = self.addProduct()
                self.isFailed = !isSuccessful
                if isSuccessful {
                    self.presentationMode.wrappedValue.dismiss()
                }
            }, label: { Text("新增") })
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color("ButtonColor"))
                .foregroundColor(Color.white)
                .cornerRadius(30)
                .font(.headline)
                .padding()
                .alert(isPresented: $isFailed) {
                    Alert(title: Text("輸入的商品資料有誤"), dismissButton: .default(Text("OK")))
                }

            Spacer()
        }
            .navigationBarTitle("新增商品")
    }

    private func addProduct() -> Bool {
        guard !name.isEmpty else {
            return false
        }
        guard let parsedPrice = Double(price) else {
            return false
        }

        let product = Product(context: context)
        product.id = UUID()
        product.name = name
        product.price = parsedPrice

        do {
            try context.save()
            return true
        } catch {
            print("\(error)")
            return false
        }
    }
}

struct AddProductView_Previews: PreviewProvider {
    static var previews: some View {
        AddProductView()
    }
}
