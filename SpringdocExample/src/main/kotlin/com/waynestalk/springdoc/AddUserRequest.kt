package com.waynestalk.springdoc

import io.swagger.v3.oas.annotations.media.Schema
import javax.validation.constraints.Min
import javax.validation.constraints.NotBlank
import javax.validation.constraints.Pattern

data class AddUserRequest(
        @field:Schema(description = "name of user", example = "Irene")
        @field:NotBlank
        val name: String,
        @field:Schema(description = "age of user", example = "18")
        @field:Min(1)
        val age: Int,
        @field:Schema(description = "email of user", example = "irene@abc.com")
        @field:Pattern(regexp = "^([a-zA-Z0-9_\\-.]+)@([a-zA-Z0-9_\\-.]+)\\.([a-zA-Z]{2,5})\$")
        val email: String)