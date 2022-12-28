import SwiftyJSON

public struct Greeting {
    public private(set) var text = "Hello, World!"

    public init() {
    }
    
    mutating public func setTextFromJSON(_ string: String) {
        guard let data = string.data(using: .utf8) else { return }
        do {
            let json = try JSON(data: data)
            text = json["text"].stringValue
        } catch {
            print(error)
        }
    }
}
