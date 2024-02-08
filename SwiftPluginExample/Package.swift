// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftPluginExample",
    platforms: [.iOS(.v13)],
    products: [
        .library(name: "SwiftPluginExample", targets: ["SwiftPluginExample"]),
    ],
    dependencies: [
        // For VersionGen example
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        
        // For Formatter example
        .package(url: "https://github.com/apple/swift-format.git", from: "509.0.0"),
    ],
    targets: [
        .target(
            name: "SwiftPluginExample",
            plugins: [
                .plugin(name: "SwiftGenPlugin"),
                .plugin(name: "VersionGenPlugin"),
            ]
        ),
        
        // For SwiftGen example
        .plugin(
            name: "SwiftGenPlugin",
            capability: .buildTool(),
            dependencies: ["swiftgen"]
        ),
        .binaryTarget(
            name: "swiftgen",
            url: "https://github.com/SwiftGen/SwiftGen/releases/download/6.6.2/swiftgen-6.6.2.artifactbundle.zip",
            checksum: "7586363e24edcf18c2da3ef90f379e9559c1453f48ef5e8fbc0b818fbbc3a045"
        ),
        
        // For VersionGen example
        .plugin(
            name: "VersionGenPlugin",
            capability: .buildTool(),
            dependencies: ["versiongen"]
        ),
        .executableTarget(
            name: "versiongen",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        
        // For Formatter example
        .plugin(
            name: "FormatterPlugin",
            capability: .command(
                intent: .sourceCodeFormatting(),
                permissions: [
                    .writeToPackageDirectory(reason: "Formatting the source files"),
                ]
            ),
            dependencies: [
                .product(name: "swift-format", package: "swift-format"),
            ]
        )
    ]
)
