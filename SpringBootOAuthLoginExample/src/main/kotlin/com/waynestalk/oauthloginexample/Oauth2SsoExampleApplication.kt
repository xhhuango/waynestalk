package com.waynestalk.oauthloginexample

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class Oauth2SsoExampleApplication

fun main(args: Array<String>) {
	runApplication<Oauth2SsoExampleApplication>(*args)
}
