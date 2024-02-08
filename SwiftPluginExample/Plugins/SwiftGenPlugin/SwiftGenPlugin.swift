import PackagePlugin

@main
struct SwiftGenPlugin: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        let configPath = context.package.directory.appending("ColorGen.yml")
        return [.prebuildCommand(
            displayName: "SwiftGen plugin",
            executable: try context.tool(named: "swiftgen").path,
            arguments: [
                "config",
                "run",
                "--verbose",
                "--config", "\(configPath)"
            ],
            environment: [
                "TARGET_DIR": target.directory,
                "DERIVED_SOURCES_DIR": context.pluginWorkDirectory
            ],
            outputFilesDirectory: context.pluginWorkDirectory
        )]
    }
}
