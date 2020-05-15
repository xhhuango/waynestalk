//
//  Person+CoreDataProperties.swift
//  CoreDataLightweightMigrationExample
//
//  Created by Wayne on 2020/5/16.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//
//

import Foundation
import CoreData


extension Person {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<Person> {
        return NSFetchRequest<Person>(entityName: "Person")
    }

    @NSManaged public var name: String?
    @NSManaged public var email: String?
    @NSManaged public var phone: String?

}
