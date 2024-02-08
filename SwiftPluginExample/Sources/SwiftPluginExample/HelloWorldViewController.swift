import UIKit

class HelloWorldViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let label = UILabel()
        label.text = "Version is \(Version.version)"
        label.textColor = Asset.blue.color
    }
}
