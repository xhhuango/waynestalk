package com.waynestalk.oauthloginexample.controller

import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController
import java.security.Principal

@RestController
class UserController {
    @GetMapping("/user")
    @PreAuthorize("hasRole('ROLE_ADMIN')")
    fun user(principal: Principal): Principal {
        return principal
    }
}