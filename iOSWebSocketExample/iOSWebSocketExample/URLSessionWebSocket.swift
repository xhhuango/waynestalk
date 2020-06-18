//
// Created by Wayne on 18/06/2020.
// Copyright (c) 2020 WaynesTalk. All rights reserved.
//

import Foundation

class URLSessionWebSocket: NSObject {
    var webSocketTask: URLSessionWebSocketTask?

    func connect() {
        guard let url = URL(string: "wss://echo.websocket.org/") else {
            print("Error: can not create URL")
            return
        }

        let urlSession = URLSession(configuration: .default, delegate: self, delegateQueue: .main)
        let request = URLRequest(url: url)

        webSocketTask = urlSession.webSocketTask(with: request)
        receive()
        webSocketTask?.resume()
    }

    private func receive() {
        webSocketTask?.receive { result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    print("Received string: \(text)")
                case .data(let data):
                    print("Received data: \(data)")
                @unknown default:
                    fatalError()
                }

            case .failure(let error):
                print(error)
            }

            self.receive()
        }
    }

    func send(message: String) {
        let message = URLSessionWebSocketTask.Message.string(message)
        webSocketTask?.send(message) { error in
            if let error = error {
                print(error)
            }
        }
    }

    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
    }
}

extension URLSessionWebSocket: URLSessionWebSocketDelegate {
    public func urlSession(_ session: URLSession,
                           webSocketTask: URLSessionWebSocketTask,
                           didOpenWithProtocol protocol: String?) {
        print("URLSessionWebSocketTask is connected")
    }

    public func urlSession(_ session: URLSession,
                           webSocketTask: URLSessionWebSocketTask,
                           didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
                           reason: Data?) {
        let reasonString: String
        if let reason = reason, let string = String(data: reason, encoding: .utf8) {
            reasonString = string
        } else {
            reasonString = ""
        }

        print("URLSessionWebSocketTask is closed: code=\(closeCode), reason=\(reasonString)")
    }
}
