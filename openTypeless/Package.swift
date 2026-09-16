// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OpenTypeless",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "OpenTypeless", targets: ["OpenTypeless"])],
    targets: [
        .executableTarget(name: "OpenTypeless"),
        .testTarget(name: "OpenTypelessTests", dependencies: ["OpenTypeless"])
    ]
)
