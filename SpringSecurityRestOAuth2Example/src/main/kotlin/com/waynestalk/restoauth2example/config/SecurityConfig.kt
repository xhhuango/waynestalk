package com.waynestalk.restoauth2example.config

import com.waynestalk.restoauth2example.auth.RestOAuth2AccessDeniedHandler
import com.waynestalk.restoauth2example.auth.RestOAuth2AuthenticationEntryPoint
import com.waynestalk.restoauth2example.auth.RestOAuth2AuthenticationFilter
import com.waynestalk.restoauth2example.auth.RestOAuth2AuthorizationFilter
import org.springframework.context.annotation.Configuration
import org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity
import org.springframework.security.config.annotation.web.builders.HttpSecurity
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter
import org.springframework.security.config.http.SessionCreationPolicy
import org.springframework.security.web.authentication.www.BasicAuthenticationFilter

@Configuration
@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true)
class SecurityConfig(
        private val restfulOAuth2AuthorizationFilter: RestOAuth2AuthorizationFilter,
        private val restfulOAuth2AuthenticationFilter: RestOAuth2AuthenticationFilter
) : WebSecurityConfigurerAdapter() {
    override fun configure(http: HttpSecurity) {
        http
                .csrf().disable()

                .addFilterBefore(restfulOAuth2AuthorizationFilter, BasicAuthenticationFilter::class.java)
                .addFilterBefore(restfulOAuth2AuthenticationFilter, BasicAuthenticationFilter::class.java)

                .sessionManagement()
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)

                .and()
                .exceptionHandling()
                .authenticationEntryPoint(RestOAuth2AuthenticationEntryPoint())
                .accessDeniedHandler(RestOAuth2AccessDeniedHandler())

                .and()
                .authorizeRequests()
                .antMatchers("/login.html").permitAll()
                .anyRequest().authenticated()
    }
}