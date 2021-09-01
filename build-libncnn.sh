#!/bin/sh

set -e

TOPDIR=$PWD

if [ ! -d ncnn ]; then
#   git clone https://github.com/Tencent/ncnn.git
    git clone https://github.com.cnpmjs.org/Tencent/ncnn.git
fi

cd $TOPDIR/ncnn
git checkout .
git checkout 20210720
cd -

rm -rf $TOPDIR/build-ncnn
mkdir -p $TOPDIR/build-ncnn
cd $TOPDIR/build-ncnn

cmake $TOPDIR/ncnn \
-DNCNN_ENABLE_LTO=ON \
-DNCNN_VULKAN=OFF \
-DNCNN_OPENMP=OFF \
-DNCNN_THREADS=OFF \
-DNCNN_PIXEL_ROTATE=OFF \
-DNCNN_PIXEL_AFFINE=OFF \
-DNCNN_PIXEL_DRAWING=OFF \
-DNCNN_DISABLE_RTTI=ON \
-DNCNN_DISABLE_EXCEPTION=ON \
-DWITH_LAYER_absval=OFF \
-DWITH_LAYER_bnll=OFF \
-DNCNN_BUILD_BENCHMARK=OFF \
-DNCNN_BUILD_EXAMPLES=OFF \
-DNCNN_BUILD_TOOLS=OFF \
-DNCNN_BUILD_TESTS=OFF \
-DNCNN_BF16=OFF \
-DNCNN_INT8=OFF \
-DNCNN_RUNTIME_CPU=OFF \
-DNCNN_AVX=OFF  \
-DNCNN_AVX2=OFF \
-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE
make -j8 && make install

rm -rf $TOPDIR/libncnn
mv $TOPDIR/build-ncnn/install/ $TOPDIR/libncnn
rm -rf $TOPDIR/build-ncnn

cd $TOPDIR
