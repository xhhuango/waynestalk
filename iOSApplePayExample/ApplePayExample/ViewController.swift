//
//  ViewController.swift
//  ApplePayExample
//
//  Created by Wayne Huang on 2024/1/16.
//

import UIKit
import PassKit

class ViewController: UIViewController {
    private let viewModel = ViewModel()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        initPaymentButton()
    }

    private func initPaymentButton() {
        var button: UIButton?
        
        if viewModel.canMakePayment() {
            button = PKPaymentButton(paymentButtonType: .buy, paymentButtonStyle: .white)
            button?.addTarget(self, action: #selector(ViewController.onPayPressed), for: .touchUpInside)
        }
        
        if let button = button {
            let constraints = [
                button.centerXAnchor.constraint(equalTo: view.centerXAnchor),
                button.centerYAnchor.constraint(equalTo: view.centerYAnchor)
            ]
            button.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(button)
            NSLayoutConstraint.activate(constraints)
        }
    }
    
    @objc private func onPayPressed(sender: AnyObject) {
        viewModel.startPayment {
            if $0 {
                debugPrint("payment has succeeded")
            }
        }
    }
}

