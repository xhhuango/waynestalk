//
//  ViewController.swift
//  iOSLocalizationExample
//
//  Created by Wayne on 2020/6/7.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    @IBOutlet weak var label: UILabel!
    
    @IBAction func onClick(_ sender: Any) {
        label.text = NSLocalizedString("Hello", comment: "Shown by click me")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
}

