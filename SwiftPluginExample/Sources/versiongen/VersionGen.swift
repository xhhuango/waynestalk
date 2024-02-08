import Foundation
import ArgumentParser

@main
struct VersionGen: ParsableCommand {
    enum Errors: Error {
        case inputPathMissing
        case outputPathMissing
        case inputFileNotFound
        case illegalInputFile
    }
    
    @Option(name: .shortAndLong, help: "The path of the version input xcconfig file")
    var input: String? = nil
    
    @Option(name: .shortAndLong, help: "The path of the version output source file")
    var output: String? = nil
    
    mutating func run() throws {
        guard let inputPath = input else {
            throw Errors.inputPathMissing
        }
        guard let outputPath = output else {
            throw Errors.outputPathMissing
        }
        
        guard FileManager.default.fileExists(atPath: inputPath) else {
            throw Errors.inputFileNotFound
        }
        
        let inputContent = try String(contentsOfFile: inputPath, encoding: .utf8)
        let inputLines = inputContent.components(separatedBy: .newlines)
        var version = ""
        for line in inputLines {
            let str = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if str.hasPrefix("VERSION") {
                let parts = str.components(separatedBy: "=")
                if parts.count == 2 {
                    version = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        }
        guard version != "" else {
            throw Errors.illegalInputFile
        }
        
        if FileManager.default.fileExists(atPath: outputPath) {
            try FileManager.default.removeItem(atPath: outputPath)
        }
        
        try """
            struct Version {
                static let version = "\(version)"
            }
            """.write(toFile: outputPath, atomically: true, encoding: .utf8)
    }
}
