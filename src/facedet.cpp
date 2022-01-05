#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ncnn/net.h>
#include "bmpfile.h"
#include "facedet.h"

#define SCORE_THRESHOLD 0.5
#define IOU_THRESHOLD   0.5

typedef struct {
    int   input_w, input_h;
    std::vector<std::vector<float>> priors;
    ncnn::Net ultraface;
} FACEDET;

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
            if (iou > threshold)  srclist[i].score = 0;
            else if (head == -1) head = i;
        }
        if (head == -1) break;
    }
}

void* facedet_init(char *paramfile, char *binfile, int inw, int inh)
{
    static const float STRIDES[] = { 8.0, 16.0, 32.0, 64.0 };
    static const float SHRINKAGES[][4] = {
        { STRIDES[0], STRIDES[1], STRIDES[2], STRIDES[3] },
        { STRIDES[0], STRIDES[1], STRIDES[2], STRIDES[3] },
    };
    static const float MINBOXES[][3] = {
        { 10.0f ,  16.0f , 24.0f  },
        { 32.0f ,  48.0f , 0      },
        { 64.0f ,  96.0f , 0      },
        { 128.0f,  192.0f, 256.0f },
    };

    FACEDET *facedet = new FACEDET();
    if (facedet) {
        facedet->input_w = inw;
        facedet->input_h = inh;

        float featuremap_size[2][4] = {
            { (float)ceil(inw / STRIDES[0]), (float)ceil(inw / STRIDES[1]), (float)ceil(inw / STRIDES[2]), (float)ceil(inw / STRIDES[3]) },
            { (float)ceil(inh / STRIDES[0]), (float)ceil(inh / STRIDES[1]), (float)ceil(inh / STRIDES[2]), (float)ceil(inh / STRIDES[3]) },
        };
        for (int n = 0; n < 4; n++) {
            float scalew = inw / SHRINKAGES[0][n];
            float scaleh = inh / SHRINKAGES[1][n];
            for (int j = 0; j < featuremap_size[1][n]; j++) {
                for (int i = 0; i < featuremap_size[0][n]; i++) {
                    float x_center = (i + 0.5) / scalew;
                    float y_center = (j + 0.5) / scaleh;
                    for (int k = 0; k < 3 && MINBOXES[n][k] != 0; k++) {
                        #define clip(x, y) ((x) < 0 ? 0 : ((x) > (y) ? (y) : (x)))
                        float w = MINBOXES[n][k] / inw;
                        float h = MINBOXES[n][k] / inh;
                        facedet->priors.push_back({ clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1) });
                    }
                }
            }
        }

        facedet->ultraface.load_param(paramfile);
        facedet->ultraface.load_model(binfile  );
    }

    return facedet;
}

void facedet_free(void *ctxt)
{
    FACEDET *facedet = (FACEDET*)ctxt;
    if (facedet) {
        facedet->ultraface.clear();
        delete facedet;
    }
}

int facedet_detect(void *ctxt, BBOX *bboxlist, int listsize, uint8_t *bitmap)
{
    if (!ctxt || !bitmap) return 0;
    FACEDET *facedet = (FACEDET*)ctxt;

    const float MEAN_VALS[3] = { 127, 127, 127 };
    const float NORM_VALS[3] = { 1.0 / 128, 1.0 / 128, 1.0 / 128 };
//  ncnn::Mat in = ncnn::Mat::from_pixels(bitmap, ncnn::Mat::PIXEL_RGBA2RGB, facedet->input_w, facedet->input_h);
    ncnn::Mat in = ncnn::Mat::from_pixels(bitmap, ncnn::Mat::PIXEL_BGR2RGB, facedet->input_w, facedet->input_h);
    in.substract_mean_normalize(MEAN_VALS, NORM_VALS);

    ncnn::Mat scores, boxes;
    ncnn::Extractor ex = facedet->ultraface.create_extractor();
    ex.set_light_mode(true);
    ex.input  ("input" , in    );
    ex.extract("scores", scores);
    ex.extract("boxes" , boxes );

    std::vector<BBOX> temp_list;
    std::vector<BBOX> bbox_list;
    for (int i = 0, n = facedet->priors.size(); i < n; i++) {
        if (scores.channel(0)[i * 2 + 1] > SCORE_THRESHOLD) {
            BBOX box;
            static const float CENTER_VARIANCE = 0.1;
            static const float SIZE_VARIANCE   = 0.2;
            float x_center = boxes.channel(0)[i * 4 + 0] * CENTER_VARIANCE * facedet->priors[i][2] + facedet->priors[i][0];
            float y_center = boxes.channel(0)[i * 4 + 1] * CENTER_VARIANCE * facedet->priors[i][3] + facedet->priors[i][1];
            float w = exp(boxes.channel(0)[i * 4 + 2] * SIZE_VARIANCE) * facedet->priors[i][2];
            float h = exp(boxes.channel(0)[i * 4 + 3] * SIZE_VARIANCE) * facedet->priors[i][3];

            box.x1 = clip(x_center - w / 2.0, 1) * facedet->input_w;
            box.y1 = clip(y_center - h / 2.0, 1) * facedet->input_h;
            box.x2 = clip(x_center + w / 2.0, 1) * facedet->input_w;
            box.y2 = clip(y_center + h / 2.0, 1) * facedet->input_h;
            box.score = clip(scores.channel(0)[i * 2 + 1], 1);
            temp_list.push_back(box);
        }
    }
    nms(bbox_list, temp_list, IOU_THRESHOLD, 0);

    int i = 0;
    if (bboxlist) {
        for (i = 0; i < (int)bbox_list.size() && i < listsize; i++) bboxlist[i] = bbox_list[i];
    }
    return i;
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
    char *paramfile= (char*)"slim_320.param";
    char *binfile  = (char*)"slim_320.bin";
    void *facedet= NULL;
    BMP   mybmp  = {0};
    BBOX  bboxes[100];
    uint32_t tick;
    int      n, i;

    if (argc >= 2) bmpfile   = argv[1];
    if (argc >= 3) paramfile = argv[2];
    if (argc >= 4) binfile   = argv[3];
    printf("bmpfile     : %s\n", bmpfile);
    printf("models files: %s, %s\n", paramfile, binfile);
    printf("\n");

    if (0 != bmp_load(&mybmp, bmpfile)) {
        printf("failed to load bmpfile %s !\n", bmpfile);
        goto done;
    }
    facedet = facedet_init(paramfile, binfile, mybmp.width, mybmp.height);

    printf("do face detection 100 times ...\n");
    tick = get_tick_count();
    for (i = 0; i < 100; i++) {
        n = facedet_detect(facedet, bboxes, 100, (uint8_t*)mybmp.pdata);
    }
    printf("finish !\n");
    printf("totoal used time: %d ms\n\n", (int)get_tick_count() - (int)tick);

    printf("face rect list:\n");
    for (i = 0; i < n; i++) {
        printf("score: %.2f, rect: (%3d %3d %3d %3d)\n", bboxes[i].score, (int)bboxes[i].x1, (int)bboxes[i].y1, (int)bboxes[i].x2, (int)bboxes[i].y2);
        bmp_rectangle(&mybmp, (int)bboxes[i].x1, (int)bboxes[i].y1, (int)bboxes[i].x2, (int)bboxes[i].y2, 0, 255, 0);
    }
    printf("\n");

    printf("save result to out.bmp\n");
    bmp_save(&mybmp, (char*)"out.bmp");
done:
    bmp_free(&mybmp);
    facedet_free(facedet);
    return 0;
}
#endif

