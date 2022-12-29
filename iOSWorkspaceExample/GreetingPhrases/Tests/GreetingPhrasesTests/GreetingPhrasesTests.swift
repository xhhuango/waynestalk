import XCTest
@testable import GreetingPhrases

final class GreetingPhrasesTests: XCTestCase {
    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(GreetingPhrases().text, "Hello, World!")
    }
}
