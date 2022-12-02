package com.waynestalk.sendingemailexample

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class SendingEmailExampleApplication

fun main(args: Array<String>) {
	runApplication<SendingEmailExampleApplication>(*args)
}
