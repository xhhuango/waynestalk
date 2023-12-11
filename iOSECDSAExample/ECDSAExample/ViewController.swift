//
//  ViewController.swift
//  ECDSAExample
//
//  Created by Wayne Huang on 2023/12/11.
//

import UIKit
import CryptoKit

class ViewController: UIViewController {
    private var privateKeyBase64 = ""
    private var publicKeyBase64 = ""
    private let message = "Hello Wayne's Talk!"
    private var signatureBase64 = ""
    
    @IBAction func generateKeys(_ sender: Any) {
        let privateKey = P256.Signing.PrivateKey()
        let publicKey = privateKey.publicKey
        
        privateKeyBase64 = privateKey.derRepresentation.base64EncodedString()
        publicKeyBase64 = publicKey.derRepresentation.base64EncodedString()
        
        print("Private Key: \(privateKeyBase64)")
        print("Public Key: \(publicKeyBase64)")
    }
    
    @IBAction func sign(_ sender: Any) {
        let privateKeyData = Data(base64Encoded: privateKeyBase64)!
        let privateKey = try! P256.Signing.PrivateKey(derRepresentation: privateKeyData)
        let signature = try! privateKey.signature(for: message.data(using: .utf8)!)
        signatureBase64 = signature.derRepresentation.base64EncodedString()
        
        print("Signature: \(signatureBase64)")
    }
    
    @IBAction func verify(_ sender: Any) {
        let publicKeyData = Data(base64Encoded: publicKeyBase64)!
        let publicKey = try! P256.Signing.PublicKey(derRepresentation: publicKeyData)
        
        var sha256 = SHA256()
        sha256.update(data: message.data(using: .utf8)!)
        let messageData = sha256.finalize()
        
        let signatureData = Data(base64Encoded: signatureBase64)!
        let signature = try! P256.Signing.ECDSASignature(derRepresentation: signatureData)
        let isValid = publicKey.isValidSignature(signature, for: messageData)
        print("Signature is \(isValid ? "valid" : "invalid")")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }


}

