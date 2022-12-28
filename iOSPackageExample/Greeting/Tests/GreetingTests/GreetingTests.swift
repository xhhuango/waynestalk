import XCTest
@testable import Greeting

final class GreetingTests: XCTestCase {
    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(Greeting().text, "Hello, World!")
    }
}
