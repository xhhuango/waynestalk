package com.waynestalk.springdoc

import io.swagger.v3.oas.annotations.OpenAPIDefinition
import io.swagger.v3.oas.annotations.info.Info
import org.springframework.context.annotation.Configuration

@OpenAPIDefinition(info = Info(title = "Wayne's Talk API", version = "v1.0.0"))
@Configuration
class SpringdocConfiguration