//
//  ContentView.swift
//  iOSStateBindingExample
//
//  Created by Wayne on 2020/6/22.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct ContentView: View {
    @State var counter = 0
    
    var body: some View {
        VStack {
            Text("Counter: \(counter)")
            CounterView(counter: $counter)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
