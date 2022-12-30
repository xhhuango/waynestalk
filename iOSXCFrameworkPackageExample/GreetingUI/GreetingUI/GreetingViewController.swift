//
//  GreetingViewController.swift
//  GreetingUI
//
//  Created by Wayne on 2022/12/30.
//

import UIKit
@_implementationOnly import Toast

public class GreetingViewController: UIViewController {
    public static func newInstance() -> GreetingViewController {
        let storyboard = UIStoryboard(name: "Greeting", bundle: Bundle(for: Self.self))
        let viewController = storyboard.instantiateViewController(withIdentifier: "GreetingViewController") as! GreetingViewController
        return viewController
    }
    
    @IBOutlet weak var greetingLabel: UILabel!
    
    @IBAction func onClick(_ sender: Any) {
        view.makeToast("Hello Wayne's Talk!")
    }
    
    public override func viewDidLoad() {
        super.viewDidLoad()

        greetingLabel.text = "Hello Wayne's Talk!"
    }
}
