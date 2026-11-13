SDK := $(shell xcrun --sdk iphoneos --show-sdk-path)
CC  := xcrun --sdk iphoneos clang
CXX := xcrun --sdk iphoneos clang++

FLAGS := -arch arm64 -isysroot $(SDK) -miphoneos-version-min=15.0 -O3 -fobjc-arc -dynamiclib
LIBS  := -framework Foundation -framework AVFoundation -framework AudioToolbox -framework Accelerate

all:
	$(CC) -arch arm64 -isysroot $(SDK) -miphoneos-version-min=15.0 -O3 -c fishhook/fishhook.c -o fishhook.o
	$(CXX) $(FLAGS) -install_name @rpath/SpatialDSP_ApplePro2_Matrix.dylib -DSPATIAL_MODE=1 -Ifishhook SpatialDSP.mm fishhook.o $(LIBS) -o SpatialDSP_ApplePro2_Matrix.dylib
	$(CXX) $(FLAGS) -install_name @rpath/SpatialDSP_ApplePro2_Front.dylib -DSPATIAL_MODE=2 -Ifishhook SpatialDSP.mm fishhook.o $(LIBS) -o SpatialDSP_ApplePro2_Front.dylib
	codesign -s - --force SpatialDSP_ApplePro2_Matrix.dylib SpatialDSP_ApplePro2_Front.dylib
