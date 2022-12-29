//
//  ViewController.swift
//  GreetingApp
//
//  Created by Wayne on 2022/12/28.
//

import UIKit
import GreetingUI
import GreetingPhrases

class ViewController: UIViewController {
    @IBOutlet weak var showButton: UIButton!
    
    @IBAction func onClick(_ sender: Any) {
        let viewController = GreetingViewController.newInstance()
        present(viewController, animated: true)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        showButton.setTitle("Show \(GreetingPhrases().text)", for: .normal)
    }
}

