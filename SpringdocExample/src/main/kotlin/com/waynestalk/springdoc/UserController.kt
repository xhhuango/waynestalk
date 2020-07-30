package com.waynestalk.springdoc

import io.swagger.v3.oas.annotations.Operation
import io.swagger.v3.oas.annotations.Parameter
import io.swagger.v3.oas.annotations.media.Content
import io.swagger.v3.oas.annotations.media.Schema
import io.swagger.v3.oas.annotations.responses.ApiResponse
import io.swagger.v3.oas.annotations.responses.ApiResponses
import org.springframework.web.bind.annotation.*
import javax.validation.Valid

@RestController
@RequestMapping("/users")
class UserController {
    var users = mutableListOf(
            User("Jack", 10, "jack@abc.com"),
            User("Monika", 11, "monika@abc.com"),
            User("Peter", 10, "peter@abc.com"),
            User("Jane", 11, "jane@abc.com")
    )

    @Operation(summary = "Find users with a given age", description = "Find users whose ages are the same as a given age")
    @GetMapping
    fun findUsers(@Parameter(description = "age to match users", example = "11") @RequestParam age: Int?) =
            if (age == null) users else users.filter { it.age == age }

    @Operation(summary = "Get a user with a given name", description = "Get a specific user with a given name")
    @ApiResponses(value = [
        ApiResponse(
                responseCode = "200",
                description = "Found user",
                content = [Content(mediaType = "application/json", schema = Schema(implementation = User::class))]
        ),
        ApiResponse(
                responseCode = "404",
                description = "User not found"
        )
    ])
    @GetMapping("/{name}")
    fun getUser(@Parameter(description = "name of user", example = "Irene") @PathVariable name: String) =
            users.find { it.name == name } ?: throw Exception("$name is not found")

    @Operation(summary = "Add a user", description = "Create a new user")
    @PostMapping
    fun addUser(@Valid @RequestBody request: AddUserRequest): User {
        val user = User(request.name, request.age, request.email)
        users.add(user)
        return user
    }
}