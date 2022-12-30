#!/bin/sh

PROJECT_NAME="GreetingUI"
OUTPUT_DIR="archives"

xcodebuild archive \
  -project "$PROJECT_NAME/$PROJECT_NAME.xcodeproj" \
  -scheme "$PROJECT_NAME" \
  -destination "generic/platform=iOS" \
  -archivePath "archives/iOS" \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES

xcodebuild archive \
  -project "$PROJECT_NAME/$PROJECT_NAME.xcodeproj" \
  -scheme "$PROJECT_NAME" \
  -destination "generic/platform=iOS Simulator" \
  -archivePath "archives/iOS-Simulator" \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES

xcodebuild -create-xcframework \
    -framework "$OUTPUT_DIR/iOS.xcarchive/Products/Library/Frameworks/$PROJECT_NAME.framework" \
    -framework "$OUTPUT_DIR/iOS-Simulator.xcarchive/Products/Library/Frameworks/$PROJECT_NAME.framework" \
    -output "$OUTPUT_DIR/$PROJECT_NAME.xcframework"

