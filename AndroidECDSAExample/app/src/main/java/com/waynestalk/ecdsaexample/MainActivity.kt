package com.waynestalk.ecdsaexample

import android.os.Bundle
import android.util.Base64
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.waynestalk.ecdsaexample.ui.theme.ECDSAExampleTheme
import java.security.KeyFactory
import java.security.KeyPairGenerator
import java.security.Signature
import java.security.interfaces.ECPrivateKey
import java.security.interfaces.ECPublicKey
import java.security.spec.ECGenParameterSpec
import java.security.spec.PKCS8EncodedKeySpec
import java.security.spec.X509EncodedKeySpec

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ECDSAExampleTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Column {
                        GeneratingKeys()
                        Sign()
                        Verify()
                    }
                }
            }
        }
    }
}

@Composable
fun GeneratingKeys() {
    Button(onClick = { generateKeys() }) {
        Text(text = "Generate Keys")
    }
}

@Composable
fun Sign() {
    Button(onClick = { sign() }) {
        Text(text = "Sign")
    }
}

@Composable
fun Verify(modifier: Modifier = Modifier) {
    Button(onClick = { verify() }, modifier = modifier) {
        Text(text = "Verify")
    }
}

var privateKeyBase64 = ""
var publicKeyBase64 = ""
var message = "Hello Wayne's Talk!"
var signatureBase64 = ""

fun generateKeys() {
    val keyPairGenerator = KeyPairGenerator.getInstance("EC")
    val ecGenParameterSpec = ECGenParameterSpec("secp256r1")
    keyPairGenerator.initialize(ecGenParameterSpec)

    val keyPair = keyPairGenerator.generateKeyPair()
    val privateKey = keyPair.private as ECPrivateKey
    val publicKey = keyPair.public as ECPublicKey

    privateKeyBase64 = Base64.encodeToString(privateKey.encoded, Base64.NO_WRAP)
    publicKeyBase64 = Base64.encodeToString(publicKey.encoded, Base64.NO_WRAP)

    println("Private Key: $privateKeyBase64")
    println("Private Key Format: ${privateKey.format}")
    println("Public Key: $publicKeyBase64")
    println("Public Key Format: ${publicKey.format}")
}

fun sign() {
    val privateKeyBytes = Base64.decode(privateKeyBase64, Base64.NO_WRAP);
    val pkcS8EncodedKeySpec = PKCS8EncodedKeySpec(privateKeyBytes)
    val keyFactory = KeyFactory.getInstance("EC")
    val privateKey = keyFactory.generatePrivate(pkcS8EncodedKeySpec)

    val signature = Signature.getInstance("SHA256withECDSA")
    signature.initSign(privateKey)

    signature.update(message.encodeToByteArray())

    val signatureBytes = signature.sign()
    signatureBase64 = Base64.encodeToString(signatureBytes, Base64.NO_WRAP)

    println("Message: $message")
    println("Signature: $signatureBase64")
}

fun verify() {
    val publicKeyBytes = Base64.decode(publicKeyBase64, Base64.NO_WRAP)
    val x509EncodedKeySpec = X509EncodedKeySpec(publicKeyBytes)
    val keyFactory = KeyFactory.getInstance("EC")
    val publicKey = keyFactory.generatePublic(x509EncodedKeySpec)

    val signature = Signature.getInstance("SHA256withECDSA")
    signature.initVerify(publicKey)

    signature.update(message.encodeToByteArray())

    val signatureBytes = Base64.decode(signatureBase64, Base64.NO_WRAP)
    val isValid = signature.verify(signatureBytes)
    println("Signature is ${if (isValid) "valid" else "inValid"}")
}