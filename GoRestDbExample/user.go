package main

import "gorm.io/gorm"

type User struct {
    gorm.Model
    Username string `json:"username"`
    Password string `json:"-"`
    Name     string `json:"name"`
    Age      uint   `json:"age"`
}
