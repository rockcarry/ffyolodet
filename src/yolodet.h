#ifndef __YOLODET_H__
#define __YOLODET_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float score, x1, y1, x2, y2;
    int   category;
} BBOX;

void* yolodet_init  (char *paramfile, char *binfile, int imgw, int imgh);
void  yolodet_free  (void *ctxt);
int   yolodet_detect(void *ctxt, BBOX *bboxlist, int n, uint8_t *bitmap);
const char* yolodet_category2str(int category);

#ifdef __cplusplus
}
#endif

#endif
