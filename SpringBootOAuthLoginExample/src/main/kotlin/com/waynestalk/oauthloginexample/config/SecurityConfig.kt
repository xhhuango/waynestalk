package com.waynestalk.oauthloginexample.config

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity
import org.springframework.security.config.annotation.web.builders.HttpSecurity
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter
import org.springframework.security.core.GrantedAuthority
import org.springframework.security.core.authority.mapping.GrantedAuthoritiesMapper
import org.springframework.security.oauth2.core.oidc.user.OidcUserAuthority

@Configuration
@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true)
class SecurityConfig : WebSecurityConfigurerAdapter() {
    override fun configure(http: HttpSecurity) {
        http
                .authorizeRequests()
                .antMatchers("/login.html").permitAll()
                .anyRequest()
                .authenticated()

                .and()
                .oauth2Login()
                .loginPage("/login.html")
                .defaultSuccessUrl("/success.html")
    }

    @Bean
    fun userAuthoritiesMapper(): GrantedAuthoritiesMapper {
        return GrantedAuthoritiesMapper { authorities ->
            val mappedAuthorities = HashSet<GrantedAuthority>()
            authorities.forEach { authority ->
                mappedAuthorities.add(authority)
                if (OidcUserAuthority::class.java.isInstance(authority)) {
                    val oidcUserAuthority = authority as OidcUserAuthority
                    val email = oidcUserAuthority.attributes["email"]

                    if (email == "your.email@waynestalk.com") {
                        mappedAuthorities.add(OidcUserAuthority("ROLE_ADMIN", oidcUserAuthority.idToken, oidcUserAuthority.userInfo))
                    }
                }
            }

            mappedAuthorities
        }
    }
}