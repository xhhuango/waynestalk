//
//  CounterView.swift
//  iOSStateBindingExample
//
//  Created by Wayne on 2020/6/22.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct CounterView: View {
    @Binding var counter: Int
    
    var body: some View {
        HStack {
            Text(String(counter))
            Button(action: { self.counter -= 1 }, label: { Text("-") })
            Button(action: { self.counter += 1 }, label: { Text("+") })
        }
            .padding()
    }
}

struct CounterView_Previews: PreviewProvider {
    static var previews: some View {
        CounterView(counter: .constant(0))
    }
}
