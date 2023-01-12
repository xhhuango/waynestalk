//
//  ViewController.swift
//  ConfigurationExample
//
//  Created by Wayne on 2023/1/11.
//

import UIKit

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let env = (Bundle.main.infoDictionary!["AppEnv"] as! String)
        print(env)
    }
}

