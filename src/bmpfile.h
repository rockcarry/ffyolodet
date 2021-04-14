#ifndef __BMPFILE_H__
#define __BMPFILE_H__

#ifdef __cplusplus
extern "C" {
#endif

/* BMP ��������Ͷ��� */
typedef struct {
    int   width;   /* ��� */
    int   height;  /* �߶� */
    int   stride;  /* ���ֽ��� */
    int   cdepth;  /* ����λ�� */
    void *pdata;   /* ָ������ */
} BMP;

int  bmp_load(BMP *pb, char *file);
int  bmp_save(BMP *pb, char *file);
void bmp_free(BMP *pb);
void bmp_setpixel (BMP *pb, int x, int y, int  r, int  g, int  b);
void bmp_getpixel (BMP *pb, int x, int y, int *r, int *g, int *b);
void bmp_rectangle(BMP *pb, int x1, int y1, int x2, int y2, int r, int g, int b);

#ifdef __cplusplus
}
#endif

#endif
