package com.waynestalk.sendingemailexample

import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.ResponseBody
import org.springframework.web.bind.annotation.RestController

@RestController
class EmailController(private val emailService: EmailService) {
    data class SendEmailRequest(
        val to: String,
        val subject: String,
        val content: String,
    )

    data class SendEmailResponse(
        val success: Boolean,
    )

    @PostMapping("/email")
    @ResponseBody
    fun sendEmail(@RequestBody request: SendEmailRequest): SendEmailResponse {
        return try {
            emailService.sendEmail(request.to, request.subject, request.content)
            SendEmailResponse(true)
        } catch (e: Exception) {
            SendEmailResponse(false)
        }
    }
}