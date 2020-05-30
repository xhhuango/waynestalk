//
//  ProductList.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/25.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct ProductListView: View {
    @Environment(\.managedObjectContext) var context

    @FetchRequest(entity: Product.entity(), sortDescriptors: [NSSortDescriptor(keyPath: \Product.name, ascending: false)])
    var products: FetchedResults<Product>

    var body: some View {
        List {
            ForEach(products, id: \.id) { product in
                NavigationLink(destination: EditProductView(product: product)) {
                    ProductRowView(product: product)
                }
            }
                .onDelete(perform: delete)
        }
            .navigationBarTitle("Product List")
            .navigationBarItems(trailing: NavigationLink(destination: AddProductView()) {
                Image(systemName: "plus")
            })
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets {
            context.delete(products[index])
        }

        do {
            try context.save()
        } catch {
            print("\(error)")
        }
    }
}

struct ProductList_Previews: PreviewProvider {
    static var previews: some View {
        ProductListView()
    }
}
