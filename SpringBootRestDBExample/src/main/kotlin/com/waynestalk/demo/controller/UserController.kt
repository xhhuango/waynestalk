package com.waynestalk.demo.controller

import com.waynestalk.demo.domain.User
import com.waynestalk.demo.repository.UserRepository
import org.springframework.http.HttpStatus
import org.springframework.web.bind.annotation.*
import org.springframework.web.server.ResponseStatusException

@RestController
@RequestMapping("/users")
class UserController(private val userRepository: UserRepository) {
    @GetMapping
    fun getUsers(@RequestParam(required = false, defaultValue = "0") age: Int): List<User> =
            if (age == 0) userRepository.findAll() else userRepository.findAllByAge(age)

    @GetMapping("/{name}")
    fun getUserBy(@PathVariable name: String) =
            userRepository.findByName(name)
                    ?: throw ResponseStatusException(HttpStatus.NOT_FOUND, "User $name not found")

    @PostMapping
    fun addUser(@RequestBody user: User) = userRepository.save(user)

    @PutMapping("/{name}")
    fun modifyUser(@PathVariable name: String, @RequestBody user: User): User {
        val found = userRepository.findByName(name)
                ?: throw ResponseStatusException(HttpStatus.NOT_FOUND, "User $name not found")
        if (user.age != null) {
            found.age = user.age
        }
        userRepository.save(found)
        return found
    }

    @DeleteMapping("/{name}")
    fun removeUser(@PathVariable name: String): User {
        val user = userRepository.findByName(name)
                ?: throw ResponseStatusException(HttpStatus.NOT_FOUND, "User $name not found")
        userRepository.delete(user)
        return user
    }
}