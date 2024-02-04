// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Greeting",
    platforms: [.iOS(.v13)],
    products: [
        .library(name: "Greeting", type: .dynamic, targets: ["Greeting"]),
    ],
    targets: [
        .target(name: "Greeting"),
        .testTarget(name: "GreetingTests", dependencies: ["Greeting"]),
    ]
)
