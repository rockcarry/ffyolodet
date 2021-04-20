#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ncnn/net.h>
#include "bmpfile.h"
#include "yolodet.h"

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

int yolodet_detect(void *ctxt, BBOX *tboxlist, int listsize, uint8_t *bitmap, int w, int h)
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

    for (i = 0; i < out.h && i < listsize; i++) {
        const float* values = out.row(i);
        tboxlist[i].category= values[0];
        tboxlist[i].score   = values[1];
        tboxlist[i].x1 = values[2] * w;
        tboxlist[i].y1 = values[3] * h;
        tboxlist[i].x2 = values[4] * w;
        tboxlist[i].y2 = values[5] * h;
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
