import PackagePlugin

@main
struct VersionGenPlugin: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        let versionConfig = context.package.directory.appending("Version.xcconfig")
        let versionSource = context.pluginWorkDirectory.appending("Version.swift")
        
        return [.buildCommand(
            displayName: "VersionGen plugin",
            executable: try context.tool(named: "versiongen").path,
            arguments: [
                "--input", "\(versionConfig)",
                "--output", "\(versionSource)",
            ],
            inputFiles: [versionConfig],
            outputFiles: [versionSource]
        )]
    }
}
