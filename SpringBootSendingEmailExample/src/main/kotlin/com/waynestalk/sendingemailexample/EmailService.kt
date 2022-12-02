package com.waynestalk.sendingemailexample

import freemarker.template.Configuration
import org.springframework.core.io.ByteArrayResource
import org.springframework.core.io.ClassPathResource
import org.springframework.mail.javamail.JavaMailSender
import org.springframework.mail.javamail.MimeMessageHelper
import org.springframework.stereotype.Service
import org.springframework.ui.freemarker.FreeMarkerTemplateUtils

@Service
class EmailService(private val javaMailSender: JavaMailSender, private val freemarkerConfig: Configuration) {
    fun sendEmail(to: String, subject: String, content: String) {
        val mimeMessage = javaMailSender.createMimeMessage()
        val helper = MimeMessageHelper(mimeMessage, true)
        helper.setFrom("Wayne's Talk <waynestalk@gmail.com>")
        helper.setTo(to)
        helper.setSubject(subject)

        val logoClassPathResource = ClassPathResource("static/logo.png")
        val logo = ByteArrayResource(logoClassPathResource.inputStream.readAllBytes())

        val parameters = mapOf(
            "logo" to "logo.png",
            "content" to content,
        )

        val template = freemarkerConfig.getTemplate("EmailTemplate.ftlh")
        val text = FreeMarkerTemplateUtils.processTemplateIntoString(template, parameters)
        helper.setText(text, true)

        // This line must be after helper.setText(), otherwise the logo won't be displayed
        helper.addInline("logo.png", logo, "image/png")

        javaMailSender.send(mimeMessage)
    }
}