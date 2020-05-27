//
//  EditProductView.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/26.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct EditProductView: View {
    @Environment(\.presentationMode) var presentationMode

    let product: Product
    
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
                let isSuccessful = self.editProduct()
                self.isFailed = !isSuccessful
                if isSuccessful {
                    self.presentationMode.wrappedValue.dismiss()
                }
            }, label: { Text("修改") })
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
            .navigationBarTitle("修改商品")
            .onAppear(perform: onAppear)
    }
    
    private func onAppear() {
        name = product.name ?? ""
        price = product.price < 0 ? "" : String(product.price)
    }

    private func editProduct() -> Bool {
        guard !name.isEmpty else {
            return false
        }
        guard let parsedPrice = Double(price) else {
            return false
        }

        product.id = UUID()
        product.name = name
        product.price = parsedPrice

        do {
            try product.managedObjectContext?.save()
            return true
        } catch {
            print("\(error)")
            return false
        }
    }
}

struct EditProductView_Previews: PreviewProvider {
    private static var product: Product {
        let product = Product()
        product.id = UUID()
        product.name = "美式咖啡"
        product.price = 25
        return product
    }
    
    static var previews: some View {
        EditProductView(product: product)
    }
}
