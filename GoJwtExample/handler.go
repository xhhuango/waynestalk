package main

import (
    "fmt"
    "github.com/gin-gonic/gin"
    "net/http"
)

func login(c *gin.Context) {
    var req struct {
        Username string `json:"username"`
        Password string `json:"password"`
    }
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{
            "error": "incorrect parameters",
        })
        return
    }

    user, err := findUserByUsername(req.Username)
    if err != nil {
        c.JSON(http.StatusNotFound, gin.H{
            "error": fmt.Sprintf("user %s not found", req.Username),
        })
        return
    }

    if user.Password != req.Password {
        c.JSON(http.StatusUnauthorized, gin.H{
            "error": "incorrect password",
        })
        return
    }

    token, err := generateToken(*user)
    if err != nil {
        c.JSON(http.StatusUnauthorized, gin.H{
            "error": err.Error(),
        })
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "token": token,
    })
}

func getUserInfo(c *gin.Context) {
    id, _, ok := getSession(c)
    if !ok {
        c.JSON(http.StatusUnauthorized, gin.H{})
        return
    }

    user, err := findUserByID(id)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{})
        return
    }

    c.JSON(http.StatusOK, user)
}
