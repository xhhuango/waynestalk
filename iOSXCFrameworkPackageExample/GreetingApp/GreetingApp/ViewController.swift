//
//  ViewController.swift
//  GreetingApp
//
//  Created by Wayne on 2022/12/30.
//

import UIKit
import GreetingUI

class ViewController: UIViewController {
    @IBAction func onClick(_ sender: Any) {
        let viewController = GreetingViewController.newInstance()
        present(viewController, animated: true)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
}

