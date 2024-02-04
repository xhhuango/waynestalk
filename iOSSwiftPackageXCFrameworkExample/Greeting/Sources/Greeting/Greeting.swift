import UIKit

public class Greeting {
    private var bundle: Bundle {
        get {
            let bundle = Bundle(for: Self.self)
            if let bundleURL = bundle.url(forResource: "Greeting_Greeting", withExtension: "bundle") {
                return Bundle(url: bundleURL) ?? bundle
            } else {
                return bundle
            }
        }
    }
    
    public func getViewController() -> UIViewController {
        let storyBoard = UIStoryboard(name: "Greeting", bundle: bundle)
        return storyBoard.instantiateInitialViewController()!
    }
}
