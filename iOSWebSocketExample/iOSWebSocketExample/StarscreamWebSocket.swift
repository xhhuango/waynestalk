//
// Created by Wayne on 18/06/2020.
// Copyright (c) 2020 WaynesTalk. All rights reserved.
//

import Foundation
import Starscream

class StarscreamWebSocket {
    var webSocket: WebSocket?

    func connect() {
        guard let url = URL(string: "wss://echo.websocket.org/") else {
            print("Error: can not create URL")
            return
        }

        let request = URLRequest(url: url)

        webSocket = WebSocket(request: request)
        webSocket?.delegate = self

        webSocket?.connect()
    }

    func send(message: String) {
        webSocket?.write(string: message)
    }

    func disconnect() {
        webSocket?.disconnect()
    }
}

extension StarscreamWebSocket: WebSocketDelegate {
    func didReceive(event: WebSocketEvent, client: WebSocket) {
        switch event {
        case .connected(_):
            print("WebSocket is connected")
        case .disconnected(let reason, let code):
            print("Disconnected: code=\(code), reason=\(reason)")
        case .text(let message):
            print("Received: \(message)")
        case .binary(_):
            break
        case .pong(_):
            break
        case .ping(_):
            break
        case .error(let error):
            print(error ?? "")
        case .viabilityChanged(_):
            break
        case .reconnectSuggested(_):
            break
        case .cancelled:
            print("WebSocket is cancelled")
        }
    }
}
