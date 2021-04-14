#ifndef __YOLODET_H__
#define __YOLODET_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   x1, y1, x2, y2;
    int   category;
    float score;
} TARGETBOX;

void* yolodet_init  (char *path);
void  yolodet_free  (void *ctxt);
int   yolodet_detect(void *ctxt, TARGETBOX *bboxlist, int n, uint8_t *bitmap, int w, int h);

#ifdef __cplusplus
}
#endif

#endif
