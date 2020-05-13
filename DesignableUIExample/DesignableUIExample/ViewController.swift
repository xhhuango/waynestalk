//
//  ViewController.swift
//  DesignableUIExample
//
//  Created by Wayne on 2020/5/12.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    @IBOutlet weak var button: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        button.layer.cornerRadius = 5
        button.layer.borderWidth = 1
        button.layer.borderColor = UIColor.green.cgColor
        
        button.backgroundColor = .red
        button.layer.shadowRadius = 5
        button.layer.shadowOpacity = 1
        button.layer.shadowOffset = .zero
        button.layer.shadowColor = UIColor.gray.cgColor
    }
}

