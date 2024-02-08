import PackagePlugin
import Foundation

@main
struct FormatterPlugin: CommandPlugin {
    func performCommand(context: PluginContext, arguments: [String]) async throws {
        let formatter = try context.tool(named: "swift-format")
        let config = context.package.directory.appending(".swift-format.json")
        
        for target in context.package.targets {
            guard let target = target as? SourceModuleTarget else { continue }

            let exec = URL(fileURLWithPath: formatter.path.string)
            let args = [
                "--configuration", "\(config)",
                "--in-place",
                "--recursive",
                "\(target.directory)"
            ]
            let process = try Process.run(exec, arguments: args)
            process.waitUntilExit()
            
            if process.terminationReason == .exit && process.terminationStatus == 0 {
                print("Formatted the source code in \(target.directory)")
            } else {
                Diagnostics.error("Formmating the source code failed: \(process.terminationReason):\(process.terminationStatus)")
            }
        }
    }
}
