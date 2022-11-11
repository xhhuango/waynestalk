package com.waynestalk.dokkaexample.core

/**
 * This class simulates car functionalities.
 *
 * @constructor Initialize a car
 * @param initialMiles Initial miles the car has driven
 * @param initialGallon Initial gallons of fuel filled into the car
 * @property milesPerGallon  A number of miles the car can drives per gallon
 */
class Car(
    initialMiles: Double = 0.0,
    initialGallon: Double = 0.0,
    val milesPerGallon: Double = 25.0,
) {
    /** The number of miles the car has driven. */
    var miles: Double = initialMiles
        private set

    /** The number of gallons the car has used. */
    var usedGallon: Double = 0.0
        private set

    /** The number of gallons of fuel in the car. */
    var gallon: Double = initialGallon
        private set

    /** Check whether the car still has fuel to drive. */
    val canDrive: Boolean
        get() = gallon > 0

    /**
     * Drive the car a given miles.
     *
     * @param miles Miles to drive
     * @return true if the car has enough fuel to drive the given miles; otherwise, false.
     * @see Car.gallon
     */
    fun drive(miles: Double): Boolean {
        require(miles >= 0)
        val requiredGallons = miles / milesPerGallon
        return if (requiredGallons > gallon) {
            usedGallon += gallon
            gallon = 0.0
            this.miles += gallon * milesPerGallon
            false
        } else {
            usedGallon += requiredGallons
            gallon -= requiredGallons
            this.miles += miles
            true
        }
    }

    /**
     * Fill a given number of gallons of fuel into the car.
     *
     * @param gallon Gallons of fuel to fill
     * @see Car.gallon
     */
    fun fillFuel(gallon: Double) {
        require(gallon >= 0)
        this.gallon += gallon
    }
}