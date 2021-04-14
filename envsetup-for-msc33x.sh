#!/bin/bash

export HOST_PLATFORM=ubuntu
export TARGET_PLATFORM=msc33x
export CC=arm-buildroot-linux-uclibcgnueabihf-gcc
export CXX=arm-buildroot-linux-uclibcgnueabihf-g++
export STRIP=arm-buildroot-linux-uclibcgnueabihf-strip
export CMAKE_TOOLCHAIN_FILE=$PWD/files/msc33x.toolchain.cmake

