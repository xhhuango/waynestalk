package main

func insertUser(user *User) error {
	if result := db.Create(user); result.Error != nil {
		return result.Error
	}
	return nil
}

func findUsers(age uint) ([]User, error) {
	var users []User
	if result := db.Where("Age > ?", age).Find(&users); result.Error != nil {
		return nil, result.Error
	}
	return users, nil
}

func findByName(name string) (*User, error) {
	var user User
	if result := db.Where(&User{Name: name}).First(&user); result.Error != nil {
		return nil, result.Error
	}
	return &user, nil
}
