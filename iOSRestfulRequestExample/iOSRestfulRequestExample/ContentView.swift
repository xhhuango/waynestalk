//
//  ContentView.swift
//  iOSRestfulRequestExample
//
//  Created by Wayne on 2020/6/18.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import SwiftUI
import Alamofire

struct ContentView: View {
    var body: some View {
        VStack {
            Button(action: { self.getRequest() }) { Text("URLSession GET") }
            Button(action: { self.postRequest() }) { Text("URLSession POST") }
            Button(action: { self.alamofireGetRequest() }) { Text("Alamofire GET") }
            Button(action: { self.alamofirePostRequest() }) { Text("Alamofire POST") }
            Button(action: { self.alamofirePostRequestAdvanced() }) { Text("Alamofire POST Advanced") }
        }
    }

    func getRequest() {
        guard let url = URL(string: "https://postman-echo.com/get?foo1=bar1&foo2=bar2") else {
            print("Error: can not create URL")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "get"

        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print(error)
                return
            }

            guard let data = data else {
                print("Did not receive data")
                return
            }

            do {
                let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
                let args = json["args"] as! [String: String]
                print(args["foo1"]!)
                print(args["foo2"]!)
            } catch {
                print("Error: can not convert data to JSON")
                return
            }
        }
        task.resume()
    }

    func postRequest() {
        guard let url = URL(string: "https://postman-echo.com/post") else {
            print("Error: can not create URL")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "post"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let data = [
            "foo1": "bar1",
            "foo2": "bar2",
        ]

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: data)
        } catch {
            print(error)
            return
        }

        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print(error)
                return
            }

            guard let data = data else {
                print("Did not receive data")
                return
            }

            do {
                let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
                let args = json["data"] as! [String: String]
                print(args["foo1"]!)
                print(args["foo2"]!)
            } catch {
                print("Error: can not convert data to JSON")
                return
            }
        }

        task.resume()
    }

    func alamofireGetRequest() {
        let parameters = [
            "foo1": "bar1",
            "foo2": "bar2",
        ]
        AF.request("https://postman-echo.com/get", method: .get, parameters: parameters)
            .responseJSON { response in
                switch response.result {
                case .success(_):
                    let json = response.value as! [String: Any]
                    let args = json["args"]! as! [String: String]
                    print(args["foo1"]!)
                    print(args["foo2"]!)

                case .failure(let error):
                    print(error)
                }
            }
    }

    func alamofirePostRequest() {
        let headers: HTTPHeaders = [
            "Content-Type": "application/json"
        ]
        let parameters = [
            "foo1": "bar1",
            "foo2": "bar2",
        ]
        AF.request("https://postman-echo.com/post", method: .post, parameters: parameters, encoding: JSONEncoding.default, headers: headers)
            .responseJSON { response in
                switch response.result {
                case .success(_):
                    let json = response.value as! [String: Any]
                    let args = json["data"]! as! [String: String]
                    print(args["foo1"]!)
                    print(args["foo2"]!)

                case .failure(let error):
                    print(error)
                }
            }
    }
    
    func alamofirePostRequestAdvanced() {
        let headers: HTTPHeaders = [
            "Content-Type": "application/json"
        ]
        let parameters = PostRequest(foo1: "bar1", foo2: "bar2")
        AF.request("https://postman-echo.com/post",
                   method: .post,
                   parameters: parameters.dictionary,
                   encoding: JSONEncoding.default,
                   headers: headers)
            .responseDecodable(of: PostResponse.self) { response in
                switch response.result {
                case .success(let value):
                    print(value.data.foo1)
                    print(value.data.foo2)

                case .failure(let error):
                    print(error)
                }
            }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

struct PostRequest: Encodable {
    let foo1: String
    let foo2: String
}

struct PostResponse: Decodable {
    let data: PostResponseData
}

struct PostResponseData: Decodable {
    let foo1: String
    let foo2: String
}

extension Encodable {
    var dictionary: [String: Any] {
        (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(self))) as? [String: Any] ?? [:]
    }
}
