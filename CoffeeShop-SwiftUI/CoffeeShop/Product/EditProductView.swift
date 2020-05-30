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
                Text("Product Name：")
                TextField("Input product name", text: $name)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
            }
                .padding()

            HStack {
                Text("Price：")
                TextField("Input price", text: $price)
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
            }, label: { Text("Edit") })
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color("ButtonColor"))
                .foregroundColor(Color.white)
                .cornerRadius(30)
                .font(.headline)
                .padding()
                .alert(isPresented: $isFailed) {
                    Alert(title: Text("Input data is incorrect"), dismissButton: .default(Text("OK")))
                }

            Spacer()
        }
            .navigationBarTitle("Edit a product")
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
        product.name = "Americano"
        product.price = 25
        return product
    }
    
    static var previews: some View {
        EditProductView(product: product)
    }
}
