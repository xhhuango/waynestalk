package main

import (
	"github.com/gin-gonic/gin"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

var db *gorm.DB

func main() {
	_db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	if err != nil {
		panic(err)
	}
	db = _db

	if err := db.AutoMigrate(&User{}); err != nil {
		panic(err)
	}

	r := gin.Default()
	r.GET("/users", ListUsers)
	r.GET("/users/:name", GetUser)
	r.POST("/users", CreateUser)

	if err := r.Run("localhost:7788"); err != nil {
		panic(err)
	}
}
