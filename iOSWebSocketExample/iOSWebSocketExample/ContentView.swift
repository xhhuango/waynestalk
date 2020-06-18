//
//  ContentView.swift
//  iOSWebSocketExample
//
//  Created by Wayne on 2020/6/18.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI

struct ContentView: View {
    let starscream = StarscreamWebSocket()
    let urlSession = URLSessionWebSocket()

    @State var text: String = "Hello WebSocket!"

    var body: some View {
        VStack {
            HStack {
                Text("Message: ")
                TextField("Message", text: $text)
            }
            Text("URLSession")
            HStack {
                Button(action: { self.urlSession.connect() }) { Text("Connect") }
                Button(action: { self.urlSession.disconnect() }) { Text("Disconnect") }
                Button(action: { self.urlSession.send(message: self.text) }) { Text("Send") }
            }
            Text("Starscream")
            HStack {
                Button(action: { self.starscream.connect() }) { Text("Connect") }
                Button(action: { self.starscream.disconnect() }) { Text("Disconnect") }
                Button(action: { self.starscream.send(message: self.text) }) { Text("Send") }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
