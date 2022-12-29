//
//  GreetingViewController.swift
//  GreetingUI
//
//  Created by Wayne on 2022/12/28.
//

import UIKit
import GreetingPhrases

public class GreetingViewController: UIViewController {
    public static func newInstance() -> GreetingViewController {
        let storyboard = UIStoryboard(name: "Greeting", bundle: Bundle(for: Self.self))
        let viewController = storyboard.instantiateViewController(withIdentifier: "GreetingViewController") as! GreetingViewController
        return viewController
    }
    
    @IBOutlet weak var greetingLabel: UILabel!
    
    public override func viewDidLoad() {
        super.viewDidLoad()

        greetingLabel.text = GreetingPhrases().text
    }
}
