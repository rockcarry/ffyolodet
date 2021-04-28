#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ncnn/net.h>
#include "bmpfile.h"
#include "yolodet.h"

#define SCORE_THRESHOLD 0.5
#define IOU_THRESHOLD   0.5

typedef struct {
    ncnn::Net dnet;
} YOLODET;

static const char* STR_CATEGORY_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};


static bool cmp_score(BBOX a, BBOX b) { return a.score > b.score; }

static void nms(std::vector<BBOX> &dstlist, std::vector<BBOX> &srclist, const float threshold, int min)
{
    if (srclist.empty()) return;
    sort(srclist.begin(), srclist.end(), cmp_score);
    int head, i;
    for (head = 0; head < (int)srclist.size(); ) {
        int x11 = srclist[head].x1;
        int y11 = srclist[head].y1;
        int x12 = srclist[head].x2;
        int y12 = srclist[head].y2;
        dstlist.push_back(srclist[head]);
        for (i = head + 1, head = -1; i < (int)srclist.size(); i++) {
            if (srclist[i].score == 0) continue;
            int x21 = srclist[i].x1;
            int y21 = srclist[i].y1;
            int x22 = srclist[i].x2;
            int y22 = srclist[i].y2;
            int xc1 = x11 > x21 ? x11 : x21;
            int yc1 = y11 > y21 ? y11 : y21;
            int xc2 = x12 < x22 ? x12 : x22;
            int yc2 = y12 < y22 ? y12 : y22;
            int sc  = (xc1 < xc2 && yc1 < yc2) ? (xc2 - xc1) * (yc2 - yc1) : 0;
            int s1  = (x12 - x11) * (y12 - y11);
            int s2  = (x22 - x21) * (y22 - y21);
            int ss  = s1 + s2 - sc;
            float iou;
            if (min) iou = (float)sc / (s1 < s2 ? s1 : s2);
            else     iou = (float)sc / ss;
            if (iou > threshold) srclist[i].score = 0;
            else if (head == -1) head = i;
        }
        if (head == -1) break;
    }
}

void* yolodet_init(char *paramfile, char *binfile)
{
    YOLODET *yolodet = new YOLODET();
    if (yolodet) {
        yolodet->dnet.load_param(paramfile);
        yolodet->dnet.load_model(binfile  );
    }
    return yolodet;
}

void yolodet_free(void *ctxt)
{
    YOLODET *yolodet = (YOLODET*)ctxt;
    if (yolodet) {
        yolodet->dnet.clear();
        delete yolodet;
    }
}

int yolodet_detect(void *ctxt, BBOX *bboxlist, int listsize, uint8_t *bitmap, int w, int h)
{
    int i;
    if (!ctxt || !bitmap) return 0;
    YOLODET *yolodet = (YOLODET*)ctxt;

    static const float MEAN_VALS[3] = { 0.f, 0.f, 0.f };
    static const float NORM_VALS[3] = { 1/255.f, 1/255.f, 1/255.f };
    ncnn::Mat in = ncnn::Mat::from_pixels(bitmap, ncnn::Mat::PIXEL_BGR2RGB, w, h);
    in.substract_mean_normalize(MEAN_VALS, NORM_VALS);

    ncnn::Mat out;
    ncnn::Extractor ex = yolodet->dnet.create_extractor();
//  ex.set_num_threads(NUMTHREADS);
    ex.set_light_mode(true);
    ex.input  ("data"  , in );
    ex.extract("output", out);
    in.release();

    std::vector<BBOX> temp_list;
    std::vector<BBOX> bbox_list;
    for (i = 0; i < out.h && i < listsize; i++) {
        const float* values = out.row(i);
        if (values[1] > SCORE_THRESHOLD) {
            BBOX box;
            box.category= values[0];
            box.score   = values[1];
            box.x1      = values[2] * w;
            box.y1      = values[3] * h;
            box.x2      = values[4] * w;
            box.y2      = values[5] * h;
            if (box.x1 < 0) box.x1 = 0;
            if (box.x1 > w) box.x1 = w - 1;
            if (box.y1 < 0) box.y1 = 0;
            if (box.y1 > h) box.y1 = h - 1;
            if (box.x2 < 0) box.x2 = 0;
            if (box.x2 > w) box.x2 = w - 1;
            if (box.y2 < 0) box.y2 = 0;
            if (box.y2 > h) box.y2 = h - 1;
            temp_list.push_back(box);
        }
    }
    nms(bbox_list, temp_list, IOU_THRESHOLD, 1);

    if (bboxlist) {
        for (i = 0; i < (int)bbox_list.size() && i < listsize; i++) bboxlist[i] = bbox_list[i];
    }
    return i;
}

const char* yolodet_category2str(int c)
{
    return (c >= 1 && c <= (int)(sizeof(STR_CATEGORY_NAMES) / sizeof(STR_CATEGORY_NAMES[0]))) ? STR_CATEGORY_NAMES[c - 1] : "unknown";
}

#ifdef _TEST_
#ifdef WIN32
#define get_tick_count GetTickCount
#else
#include <time.h>
static uint32_t get_tick_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
#endif

int main(int argc, char *argv[])
{
    char *bmpfile  = (char*)"test.bmp";
    char *paramfile= (char*)"yolo-fastest-1.1-xl.param";
    char *binfile  = (char*)"yolo-fastest-1.1-xl.bin";
    void *yolodet  = NULL;
    BMP   mybmp    = {0};
    BBOX  tboxes[100];
    uint32_t tick;
    int      n, i;

    if (argc >= 2) bmpfile   = argv[1];
    if (argc >= 3) paramfile = argv[2];
    if (argc >= 4) binfile   = argv[3];
    printf("bmpfile  : %s\n", bmpfile  );
    printf("paramfile: %s\n", paramfile);
    printf("binfile  : %s\n", binfile  );
    printf("\n");

    yolodet = yolodet_init(paramfile, binfile);
    if (0 != bmp_load(&mybmp, bmpfile)) {
        printf("failed to load bmpfile %s !\n", bmpfile);
        goto done;
    }

    printf("do face detection 100 times ...\n");
    tick = get_tick_count();
    for (i = 0; i < 100; i++) {
        n = yolodet_detect(yolodet, tboxes, 100, (uint8_t*)mybmp.pdata, mybmp.width, mybmp.height);
    }
    printf("finish !\n");
    printf("totoal used time: %d ms\n\n", (int)get_tick_count() - (int)tick);

    printf("target rect list:\n");
    for (i = 0; i < n; i++) {
        printf("score: %.2f, category: %12s, rect: (%3d %3d %3d %3d)\n", tboxes[i].score, yolodet_category2str(tboxes[i].category), tboxes[i].x1, tboxes[i].y1, tboxes[i].x2, tboxes[i].y2);
        bmp_rectangle(&mybmp, tboxes[i].x1, tboxes[i].y1, tboxes[i].x2, tboxes[i].y2, 0, 255, 0);
    }
    printf("\n");

    printf("save result to out.bmp\n");
    bmp_save(&mybmp, (char*)"out.bmp");

done:
    bmp_free(&mybmp);
    yolodet_free(yolodet);
    return 0;
}
#endif
