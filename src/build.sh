#!/bin/sh

set -e

CXX_FLAGS="-I$PWD/../libncnn/include -std=c++11 -Ofast -ffunction-sections -fdata-sections -fPIC"
LD_FLAGS="-L$PWD/../libncnn/lib -lncnn -lpthread -flto -Wl,-gc-sections -Wl,-strip-all"

case "$1" in
"")
    $CXX -Wall -D_TEST_ bmpfile.c yolodet.cpp $CXX_FLAGS $LD_FLAGS -o test
    case "$TARGET_PLATFORM" in
    win32)
        $CXX --shared yolodet.cpp $CXX_FLAGS $LD_FLAGS -o yolodet.dll
        dlltool -l yolodet.lib -d yolodet.def
        $STRIP *.exe *.dll
        ;;
    ubuntu)
        $CXX --shared yolodet.cpp $CXX_FLAGS $LD_FLAGS -o libyolodet.so
        $STRIP test *.so
        ;;
    msc33x)
        $CXX --shared yolodet.cpp $CXX_FLAGS $LD_FLAGS -o libyolodet.so
        $STRIP test *.so
#       $CXX -Wall -c bmpfile.c yolodet.cpp $CXX_FLAGS $LD_FLAGS
#       cp $PWD/../libncnn/lib/libncnn.a libyolodet.a
#       arm-buildroot-linux-uclibcgnueabihf-gcc-ar rcs libyolodet.a bmpfile.o yolodet.o
        ;;
    esac
    ;;
clean)
    rm -rf test *.so *.dll *.exe out.bmp
    ;;
esac
