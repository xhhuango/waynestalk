//
//  Product+CoreDataProperties.swift
//  CoffeeShop
//
//  Created by Wayne on 2020/5/26.
//  Copyright © 2020 WaynesTalk. All rights reserved.
//
//

import Foundation
import CoreData


extension Product: Identifiable {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<Product> {
        return NSFetchRequest<Product>(entityName: "Product")
    }

    @NSManaged public var id: UUID?
    @NSManaged public var name: String?
    @NSManaged public var price: Double

}
