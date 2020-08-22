package com.waynestalk.jwtexample.config

import io.swagger.v3.oas.models.Components
import io.swagger.v3.oas.models.OpenAPI
import io.swagger.v3.oas.models.info.Info
import io.swagger.v3.oas.models.security.SecurityRequirement
import io.swagger.v3.oas.models.security.SecurityScheme
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class SwaggerConfiguration {
    @Bean
    fun customOpenAPI(): OpenAPI {
        val securitySchemeName = "Auth JWT"
        return OpenAPI()
                .addSecurityItem(SecurityRequirement().addList(securitySchemeName))
                .components(
                        Components()
                                .addSecuritySchemes(securitySchemeName,
                                        SecurityScheme()
                                                .name(securitySchemeName)
                                                .type(SecurityScheme.Type.HTTP)
                                                .scheme("bearer")
                                                .bearerFormat("JWT")
                                )
                )
                .info(Info().title("Wayne's Talk APIs").version("v1.0.0"))
    }
}