package com.waynestalk.jwtexample.controller

import com.waynestalk.jwtexample.repository.MemberRepository
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController
import javax.servlet.http.HttpServletRequest

@RestController
class MemberController(private val memberRepository: MemberRepository) {
    @GetMapping("/greet")
    @PreAuthorize("hasRole('ROLE_ADMIN')")
    fun greet(request: HttpServletRequest): GreetResponse {
        val member = memberRepository.findByUsername(request.userPrincipal.name)!!
        return GreetResponse("Hello ${member.name}")
    }

    data class GreetResponse(val message: String)
}