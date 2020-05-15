//
//  PersonManager.swift
//  CoreDataLightweightMigrationExample
//
//  Created by Wayne on 2020/5/16.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//

import Foundation
import CoreData

class PersonManager {
    static let shared = PersonManager()

    private let persistentContainer: NSPersistentContainer
    private let managedObjectContext: NSManagedObjectContext

    init() {
        persistentContainer = NSPersistentContainer(name: "CoreDataLightweightMigrationExample")
        let description = persistentContainer.persistentStoreDescriptions[0]
        description.shouldMigrateStoreAutomatically = true
        description.shouldInferMappingModelAutomatically = true
        persistentContainer.persistentStoreDescriptions =  [description]

        persistentContainer.loadPersistentStores { (storeDescription, error) in
            if let error = error {
                fatalError("Error loading store \(storeDescription), \(error)")
            }
        }
        
        managedObjectContext = persistentContainer.newBackgroundContext()
    }

    func add(name: String, email: String, phone: String) {
        let person = Person(context: managedObjectContext)
        person.name = name
        person.email = email
        person.phone = phone

        do {
            try managedObjectContext.save()
        } catch {
            print(error)
        }

    }

    func read() -> [Person]? {
        let request: NSFetchRequest<Person> = Person.fetchRequest()
        do {
            return try managedObjectContext.fetch(request)
        } catch {
            print(error)
            return nil
        }
    }
}
