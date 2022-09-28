#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ncnn/net.h>
#include "bmpfile.h"
#include "yolodet.h"

#define ENABLE_RESIZE_INPUT  0
#define RESIZE_INPUT_WIDTH   256
#define RESIZE_INPUT_HEIGHT  160

#define ALIGN(a, b) (((a) + (b) - 1) & ~((b) - 1))

typedef struct {
    ncnn::Net dnet;
    int   modelver;
    int   imagew, imageh;
    int   inputw, inputh;
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

void* yolodet_init(char *paramfile, char *binfile, int imgw, int imgh)
{
    YOLODET *yolodet = new YOLODET();
    if (yolodet) {
        yolodet->dnet.load_param(paramfile);
        yolodet->dnet.load_model(binfile  );
        yolodet->modelver = strstr(paramfile, "fastest-v2") ? 2 : 1;
        yolodet->imagew   = imgw;
        yolodet->imageh   = imgh;
        yolodet->inputw   = ENABLE_RESIZE_INPUT ? RESIZE_INPUT_WIDTH : imgw;
        yolodet->inputh   = ENABLE_RESIZE_INPUT ? RESIZE_INPUT_HEIGHT: imgh;
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

static int bbox_cmp(const void *p1, const void *p2)
{
    if      (((BBOX*)p1)->score < ((BBOX*)p2)->score) return  1;
    else if (((BBOX*)p1)->score > ((BBOX*)p2)->score) return -1;
    else return 0;
}

static int nms(BBOX *bboxlist, int n, float threshold, int min)
{
    int i, j, c;
    if (!bboxlist || !n) return 0;
    qsort(bboxlist, n, sizeof(BBOX), bbox_cmp);
    for (i=0; i<n && i!=-1; ) {
        for (c=i,j=i+1,i=-1; j<n; j++) {
            if (bboxlist[j].score == 0) continue;
            if (bboxlist[c].category == bboxlist[j].category) {
                float xc1, yc1, xc2, yc2, sc, s1, s2, ss, iou;
                xc1 = bboxlist[c].x1 > bboxlist[j].x1 ? bboxlist[c].x1 : bboxlist[j].x1;
                yc1 = bboxlist[c].y1 > bboxlist[j].y1 ? bboxlist[c].y1 : bboxlist[j].y1;
                xc2 = bboxlist[c].x2 < bboxlist[j].x2 ? bboxlist[c].x2 : bboxlist[j].x2;
                yc2 = bboxlist[c].y2 < bboxlist[j].y2 ? bboxlist[c].y2 : bboxlist[j].y2;
                sc  = (xc1 < xc2 && yc1 < yc2) ? (xc2 - xc1) * (yc2 - yc1) : 0;
                s1  = (bboxlist[c].x2 - bboxlist[c].x1) * (bboxlist[c].y2 - bboxlist[c].y1);
                s2  = (bboxlist[j].x2 - bboxlist[j].x1) * (bboxlist[j].y2 - bboxlist[j].y1);
                ss  = s1 + s2 - sc;
                if (min) iou = sc / (s1 < s2 ? s1 : s2);
                else     iou = sc / ss;
                if (iou > threshold) bboxlist[j].score = 0;
                else if (i == -1) i = j;
            } else if (i == -1) i = j;
        }
    }
    for (i=0,j=0; i<n; i++) {
        if (bboxlist[i].score) {
            bboxlist[j  ].score    = bboxlist[i].score;
            bboxlist[j  ].category = bboxlist[i].category;
            bboxlist[j  ].x1       = bboxlist[i].x1;
            bboxlist[j  ].y1       = bboxlist[i].y1;
            bboxlist[j  ].x2       = bboxlist[i].x2;
            bboxlist[j++].y2       = bboxlist[i].y2;
        }
    }
    return j;
}

#define ANCHOR_NUM    3
#define CATEGORY_NUM  80
#define SCORE_THRESH  0.3
#define NMSIOU_THRESH 0.5
static float s_v2_anchor_boxes[] = { 12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87 };
static int gen_bbox(int inputw, int inputh, ncnn::Mat *out, int n, BBOX *bboxlist, int listsize)
{
    int ow, oh, oc, i, j, k, l, num = 0;
    while (--n >= 0) {
        ow = out[n].h;
        oh = out[n].c;
        oc = out[n].w;

        for (i=0; i<oh; i++) {
            float *values = out[n].channel(i);
            for (j=0; j<ow; j++) {
                for (k=0; k<ANCHOR_NUM; k++) {
                    float objscore = values[4 * ANCHOR_NUM + k];
                    float clsscore = values[4 * ANCHOR_NUM + ANCHOR_NUM + 0];
                    int   clsidx   = 0;
                    for (l=1; l<CATEGORY_NUM; l++) {
                        if (clsscore < values[4 * ANCHOR_NUM + ANCHOR_NUM + l]) {
                            clsscore = values[4 * ANCHOR_NUM + ANCHOR_NUM + l];
                            clsidx   = l;
                        }
                    }
                    if (objscore * clsscore >= SCORE_THRESH) {
                        float bcx = ((values[k * 4 + 0] * 2 - 0.5) + j) / ow;
                        float bcy = ((values[k * 4 + 1] * 2 - 0.5) + i) / oh;
                        float bw  = pow((values[k * 4 + 2] * 2), 2) * s_v2_anchor_boxes[(n * ANCHOR_NUM * 2) + k * 2 + 0];
                        float bh  = pow((values[k * 4 + 3] * 2), 2) * s_v2_anchor_boxes[(n * ANCHOR_NUM * 2) + k * 2 + 1];
                        if (num < listsize) {
                            bboxlist[num].category = clsidx + 1;
                            bboxlist[num].score    = objscore * clsscore;
                            bboxlist[num].x1       = bcx - bw / inputw * 0.5f;
                            bboxlist[num].y1       = bcy - bh / inputh * 0.5f;
                            bboxlist[num].x2       = bcx + bw / inputw * 0.5f;
                            bboxlist[num].y2       = bcy + bh / inputh * 0.5f;
                            num++;
                        }
                    }
                }
                values += oc;
            }
        }
    }
    return nms(bboxlist, num, NMSIOU_THRESH, 1);
}

int yolodet_detect(void *ctxt, BBOX *bboxlist, int listsize, uint8_t *bitmap)
{
    int i;
    if (!ctxt || !bitmap) return 0;
    YOLODET *yolodet = (YOLODET*)ctxt;

    static const float MEAN_VALS[3] = { 0.f, 0.f, 0.f };
    static const float NORM_VALS[3] = { 1/255.f, 1/255.f, 1/255.f };

//  ncnn::Mat in = ncnn::Mat::from_pixels_resize(bitmap, ncnn::Mat::PIXEL_RGBA2RGB, yolodet->imagew, yolodet->imageh, yolodet->inputw, yolodet->inputh);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bitmap, ncnn::Mat::PIXEL_BGR2RGB, yolodet->imagew, yolodet->imageh, yolodet->inputw, yolodet->inputh);
    in.substract_mean_normalize(MEAN_VALS, NORM_VALS);

    ncnn::Mat out[2];
    ncnn::Extractor ex = yolodet->dnet.create_extractor();
    ex.set_light_mode(true);

    switch (yolodet->modelver) {
    case 1:
        ex.input  ("data"  , in    );
        ex.extract("output", out[0]);
        for (i = 0; i < out[0].h && i < listsize; i++) {
            const float * values = out[0].row(i);
            bboxlist[i].category = values[0];
            bboxlist[i].score    = values[1];
            bboxlist[i].x1       = values[2];
            bboxlist[i].y1       = values[3];
            bboxlist[i].x2       = values[4];
            bboxlist[i].y2       = values[5];
        }
        break;
    case 2:
        ex.input("input.1", in);
        ex.extract("794", out[0]);
        ex.extract("796", out[1]);
        i = gen_bbox(yolodet->inputw, yolodet->inputh, out, 2, bboxlist, listsize);
        break;
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
    BBOX  bboxes[100];
    uint32_t tick;
    int      n, i;

    if (argc >= 2) bmpfile   = argv[1];
    if (argc >= 3) paramfile = argv[2];
    if (argc >= 4) binfile   = argv[3];
    printf("bmpfile  : %s\n", bmpfile  );
    printf("paramfile: %s\n", paramfile);
    printf("binfile  : %s\n", binfile  );
    printf("\n");

    if (0 != bmp_load(&mybmp, bmpfile)) {
        printf("failed to load bmpfile %s !\n", bmpfile);
        goto done;
    }

    yolodet = yolodet_init(paramfile, binfile, mybmp.width, mybmp.height);
    printf("do face detection 100 times ...\n");
    tick = get_tick_count();
    for (i = 0; i < 100; i++) {
        n = yolodet_detect(yolodet, bboxes, 100, (uint8_t*)mybmp.pdata);
    }
    printf("finish !\n");
    printf("totoal used time: %d ms\n\n", (int)get_tick_count() - (int)tick);

    printf("target rect list:\n");
    for (i = 0; i < n; i++) {
        printf("score: %.2f, category: %12s, rect: (%3d %3d %3d %3d)\n", bboxes[i].score, yolodet_category2str(bboxes[i].category),
            (int)(bboxes[i].x1 * mybmp.width), (int)(bboxes[i].y1 * mybmp.height), (int)(bboxes[i].x2 * mybmp.width), (int)(bboxes[i].y2 * mybmp.height));
        bmp_rectangle(&mybmp, (int)(bboxes[i].x1 * mybmp.width), (int)(bboxes[i].y1 * mybmp.height), (int)(bboxes[i].x2 * mybmp.width), (int)(bboxes[i].y2 * mybmp.height), 0, 255, 0);
    }
    printf("\n");

    printf("save result to out.bmp\n");
    bmp_save(&mybmp, (char*)"out.bmp");

done:
    yolodet_free(yolodet);
    bmp_free(&mybmp);
    return 0;
}
#endif
