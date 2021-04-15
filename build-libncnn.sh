#!/bin/sh

set -e

TOPDIR=$PWD

if [ ! -d ncnn ]; then
#   git clone https://github.com/Tencent/ncnn.git
    git clone https://github.com.cnpmjs.org/Tencent/ncnn.git
fi

cd $TOPDIR/ncnn
git checkout 20210322
git checkout .
echo "target_compile_options(ncnn PUBLIC -ffunction-sections -fdata-sections)" >> $TOPDIR/ncnn/src/CMakeLists.txt
cd -

rm -rf $TOPDIR/build-ncnn
mkdir -p $TOPDIR/build-ncnn
cd $TOPDIR/build-ncnn

cmake $TOPDIR/ncnn \
-DNCNN_DISABLE_RTTI=ON \
-DNCNN_DISABLE_EXCEPTION=ON \
-DNCNN_VULKAN=OFF \
-DNCNN_PIXEL_ROTATE=OFF \
-DNCNN_PIXEL_AFFINE=OFF \
-DNCNN_OPENMP=OFF \
-DWITH_LAYER_absval=OFF \
-DWITH_LAYER_bnll=OFF \
-DNCNN_ENABLE_LTO=ON \
-DNCNN_BUILD_BENCHMARK=OFF \
-DNCNN_BUILD_EXAMPLES=OFF \
-DNCNN_BUILD_TOOLS=OFF \
-DNCNN_BUILD_TESTS=OFF \
-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE
make -j8 && make install

rm -rf $TOPDIR/libncnn
mv $TOPDIR/build-ncnn/install/ $TOPDIR/libncnn
rm -rf $TOPDIR/build-ncnn

cd $TOPDIR
