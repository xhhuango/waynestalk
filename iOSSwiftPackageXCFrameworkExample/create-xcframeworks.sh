#!/bin/bash

SIMULATOR_SDK="iphonesimulator"
DEVICE_SDK="iphoneos"

PACKAGE="Greeting"
CONFIGURATION="Release"
DEBUG_SYMBOLS="true"

VERSION=$(sed -n 's/VERSION = \([\w]*\)/\1/p' Version.xcconfig | head -n 1 | tr -d ';' | tr -s '[:blank:]' | tr -d "\t")
BUILD=$(sed -n 's/BUILD = \([\w]*\)/\1/p' Version.xcconfig | head -n 1 | tr -d ';' | tr -s '[:blank:]' | tr -d "\t")

BUILD_DIR="$(pwd)/build"
DIST_DIR="$(pwd)/dist"

build_framework() {
  scheme=$1
  sdk=$2
  if [ "$2" = "$SIMULATOR_SDK" ]; then
    dest="generic/platform=iOS Simulator"
  elif [ "$2" = "$DEVICE_SDK" ]; then
    dest="generic/platform=iOS"
  else
    echo "Unknown SDK $2"
    exit 11
  fi

  echo "Build framework"
  echo "Scheme: $scheme"
  echo "Configuration: $CONFIGURATION"
  echo "SDK: $sdk"
  echo "Destination: $dest"
  echo

  (cd "$PACKAGE" &&
    xcodebuild \
      -scheme "$scheme" \
      -configuration "$CONFIGURATION" \
      -destination "$dest" \
      -sdk "$sdk" \
      -derivedDataPath "$BUILD_DIR" \
      SKIP_INSTALL=NO \
      BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
      OTHER_SWIFT_FLAGS="-no-verify-emitted-module-interface") || exit 12

  product_path="$BUILD_DIR/Build/Products/$CONFIGURATION-$sdk"
  framework_path="$BUILD_DIR/Build/Products/$CONFIGURATION-$sdk/PackageFrameworks/$scheme.framework"

  # Copy Headers
  headers_path="$framework_path/Headers"
  mkdir "$headers_path"
  cp -pv \
    "$BUILD_DIR/Build/Intermediates.noindex/$PACKAGE.build/$CONFIGURATION-$sdk/$scheme.build/Objects-normal/arm64/$scheme-Swift.h" \
    "$headers_path/" || exit 13

  # Copy other headers from Sources/
  headers=$(find "$PACKAGE/$scheme" -name "*.h")
  for h in $headers; do
    cp -pv "$h" "$headers_path" || exit 14
  done

  # Copy Modules
  modules_path="$framework_path/Modules"
  mkdir "$modules_path"
  cp -pv \
    "$BUILD_DIR/Build/Intermediates.noindex/$PACKAGE.build/$CONFIGURATION-$sdk/$scheme.build/$scheme.modulemap" \
    "$modules_path" || exit 15
  mkdir "$modules_path/$scheme.swiftmodule"
  cp -pv "$product_path/$scheme.swiftmodule"/*.* "$modules_path/$scheme.swiftmodule/" || exit 16

  # Copy Bundle
  bundle_dir="$product_path/${PACKAGE}_$scheme.bundle"
  if [ -d "$bundle_dir" ]; then
    cp -prv "$bundle_dir"/* "$framework_path/" || exit 17
  fi
}

create_xcframework() {
  scheme=$1

  echo "Create $scheme.xcframework"

  args=""
  shift 1
  for p in "$@"; do
    args+=" -framework $BUILD_DIR/Build/Products/$CONFIGURATION-$p/PackageFrameworks/$scheme.framework"
    if [ "$DEBUG_SYMBOLS" = "true" ]; then
      args+=" -debug-symbols $BUILD_DIR/Build/Products/$CONFIGURATION-$p/$scheme.framework.dSYM"
    fi
  done

  xcodebuild -create-xcframework $args -output "$DIST_DIR/$scheme.xcframework" || exit 21
}

reset_package_type() {
  (cd "$PACKAGE" && sed -i '' 's/\( type: .dynamic,\)//g' Package.swift) || exit
}

set_package_type_as_dynamic() {
  (cd "$PACKAGE" && sed -i '' "s/\(.library(name: *\"$1\",\)/\1 type: .dynamic,/g" Package.swift) || exit
}

echo "**********************************"
echo "******* Build XCFrameworks *******"
echo "**********************************"
echo

rm -rf "$BUILD_DIR"
rm -rf "$DIST_DIR"

reset_package_type

set_package_type_as_dynamic "Greeting"
build_framework "Greeting" "$SIMULATOR_SDK"
build_framework "Greeting" "$DEVICE_SDK"
create_xcframework "Greeting" "$SIMULATOR_SDK" "$DEVICE_SDK"
