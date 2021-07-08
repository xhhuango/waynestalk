package main

func insertUser(username string, password string, age uint) (*User, error) {
    user := User{
        Username: username,
        Password: password,
        Age:      age,
    }
    if res := DB.Create(&user); res.Error != nil {
        return nil, res.Error
    }
    return &user, nil
}

func findUserByUsername(username string) (*User, error) {
    var user User
    if res := DB.Where("username = ?", username).Find(&user); res.Error != nil {
        return nil, res.Error
    }
    return &user, nil
}

func findUserByID(id uint) (*User, error) {
    var user User
    if res := DB.Find(&user, id); res.Error != nil {
        return nil, res.Error
    }
    return &user, nil
}
