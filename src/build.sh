#!/bin/sh

set -e

CXX_FLAGS="-I$PWD/../libncnn/include -std=c++11 -Ofast -ffunction-sections -fdata-sections -fPIC"
LD_FLAGS="-L$PWD/../libncnn/lib -lncnn -lpthread -flto -Wl,-gc-sections -Wl,-strip-all"

case "$1" in
"")
    $CXX -Wall -D_TEST_ bmpfile.c yolodet.cpp $CXX_FLAGS $LD_FLAGS -o yolotest
    $CXX -Wall -D_TEST_ bmpfile.c facedet.cpp $CXX_FLAGS $LD_FLAGS -o facetest
    $CXX -Wall -c bmpfile.c yolodet.cpp facedet.cpp $CXX_FLAGS $LD_FLAGS
    cp $PWD/../libncnn/lib/libncnn.a libyolodet.a
    ${CC}-ar rcs libyolodet.a bmpfile.o yolodet.o facedet.o
    ;;
clean)
    rm -rf yolotest facetest *.a *.o *.exe out.bmp
    ;;
esac
