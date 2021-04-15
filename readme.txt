+--------------------------------+
 基于 ncnn + yolofast 的目标检测
+--------------------------------+

Yolo-Fastest:
https://github.com/dog-qiuqiu/Yolo-Fastest

yolo 神经网络的目标检测和分类，并且提供了可用于 ncnn 的模型文件
因此我们可以非常方便的移植到 ncnn 上

本程序基于腾讯的 ncnn + yolofast 模型，实现了目标检测和分类功能
可以识别出 82 种目标在图像中的位置，以及目标的分类（包括了人脸、人体、猫猫狗狗、汽车、各种动物...）

在 ubuntu、msys2 和 msc33x 嵌入式 linux 平台都可以编译通过和使用
（在 msc33x 平台实测，检测 312x168 的图像需要 370ms，还需要优化否则实用性不大）


编译和运行
----------
目前已经在三个平台上编译通过：
1. ubuntu - envsetup-for-ubuntu.sh
2. msys2  - envsetup-for-msys2.sh
3. msc33x - envsetup-for-msc33x.sh

编译时需要首先执行对应的 envsetup-xxx.sh 设定环境变量：
source envsetup-for-msys2.sh

编译 libncnn 库：
./build-libncnn.sh

编译 src 下的源代码：
cd src
./build.sh

最终生成 test 程序，可用于目标检测

./test test.bmp yolo-fastest-1.1.param yolo-fastest-1.1.bin
第一个参数是要检测的图片文件
第二个参数是模型 param 文件路径
第二个参数是模型 bin 文件路径



rockcarry
2021-4-14

