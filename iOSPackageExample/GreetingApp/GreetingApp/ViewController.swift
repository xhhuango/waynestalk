//
//  ViewController.swift
//  GreetingApp
//
//  Created by Wayne on 2022/12/26.
//

import UIKit
import Toast
import Greeting

class ViewController: UIViewController {
    @IBOutlet weak var greetingTextField: UITextField!
    @IBOutlet weak var greetingLabel: UILabel!
    
    @IBAction func onClick(_ sender: Any) {
        let text = greetingTextField.text ?? ""
        greeting.setTextFromJSON("{\"text\":\"\(text)\"}")
        greetingLabel.text = greeting.text
        view.makeToast(greeting.text)
    }
    
    private var greeting = Greeting()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        greetingLabel.text = greeting.text
    }
}

