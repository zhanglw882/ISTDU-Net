import numpy as np
import cv2

def imageClone(src, dst, p, pixMax=None, mul=None, mask=None, OSTU=False, Gauss=False, blend=None):
    '''
    .   @param src Input 8-bit 1-channel image. Target
    .   @param dst Input 8-bit 1-channel image. Background
    .   @param mask Input 8-bit 1-channel image.
    .   @param p Point in dst image where object is placed.
    '''
    h, w = dst.shape
    xCenter, yCenter = p
    if xCenter == -1 and yCenter == -1:
        return False, cv2.GaussianBlur(dst.copy(), (3, 3), 0) if Gauss else dst.copy(), np.zeros(dst.shape, dtype=np.bool_)
    assert xCenter >= 0 and xCenter < w and yCenter >= 0 and yCenter < h
    m, n = src.shape
    xLeft, yTop = xCenter-(n-1)//2, yCenter-(m-1)//2
    xRight, yDown = xLeft + n, yTop + m
    srcLeft, srcRight, srcTop, srcDown = 0, n, 0, m
    if xLeft < 0:
        srcLeft = -xLeft
        xLeft = 0
    if xRight > w:
        srcRight = -xRight+w
        xRight = w
    if yTop < 0:
        srcTop = -yTop
        yTop = 0
    if yDown > h:
        srcDown = -yDown+h
        yDown = h

    if not mask: mask = 1

    if OSTU:
        t1, thd= cv2.threshold(src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        maskAux = cv2.connectedComponents(thd)
        maskAux = maskAux[1]
        maskAux[maskAux[0,0]==maskAux] = 0
        maskAux[maskAux[0,-1]==maskAux] = 0
        maskAux[maskAux[-1,0]==maskAux] = 0
        maskAux[maskAux[-1,-1]==maskAux] = 0
        mask = maskAux>0

    nomalMax = np.max(src) if not pixMax else pixMax
    nomalMax = int(nomalMax*mul) if mul else nomalMax
    # src = cv2.normalize(src, None, 0, nomalMax, cv2.NORM_MINMAX)
    # src = cv2.equalizeHist(src)
    src = cv2.normalize(src, None, 0, nomalMax, cv2.NORM_MINMAX)


    src *= mask

    target = src[srcTop:srcDown, srcLeft:srcRight]

    # backPitch = dst[yTop:yDown, xLeft:xRight]
    # targetMax = np.max(target)
    # mTarget, nTarget = target.shape

    tmp = np.zeros((h, w))
    tmp[yTop:yDown, xLeft:xRight] = target

    tmp[tmp>255] = 255
    tmp = np.uint8(tmp)

    maskRet = tmp > dst

    # ok = np.sum(mask) > 3
    # ok = np.max(target) > np.mean(dst[srcTop:srcDown, srcLeft:srcRight]) + 5

    ret = dst.copy()
    ok = False
    if np.sum(maskRet):
        # ok = np.mean(ret[maskRet])+10 < np.max(tmp[maskRet])
        ok = np.max(ret[maskRet])+15 < np.max(tmp[maskRet])

    if ok:
        ret[maskRet] = tmp[maskRet]

    if Gauss:
        ret = cv2.GaussianBlur(ret, (3, 3), 0)

    # return ok, ret, mask
    return ok, ret, maskRet if ok else np.zeros(maskRet.shape, dtype=np.bool_)
    # return ret
    # return mask*1.0

if __name__ == '__main__':
    t = '/media/dell/56FE4D96FE4D6F75/ZHL/ds/205sds/changing/target1/atarget/00000.png'
    # t = '/media/dell/56FE4D96FE4D6F75/ZHL/ds/205sds/changing/target1/atarget/00000.png'
    # t = '/media/dell/56FE4D96FE4D6F75/ZHL/program/centernet/Targets/tt/4.png'
    b = '/media/dell/56FE4D96FE4D6F75/ZHL/ds/cus/1/im001593.png'
    # b = '/media/dell/56FE4D96FE4D6F75/ZHL/ds/205sds/changing/target4/n/im000004.png'

    tt = cv2.imread(t, cv2.IMREAD_GRAYSCALE)
    bb = cv2.imread(b, cv2.IMREAD_GRAYSCALE)

    tt[0:10,0:3] = 255

    # tt = cv2.imread(t, cv2.IMREAD_COLOR)
    # bb = cv2.imread(b, cv2.IMREAD_COLOR)
    p = (bb.shape[1]//2, bb.shape[0]//2-100)
    # p = [0,0]
    # p = [599,799]
    # p = [799,599]
    # p = [800,600]
    # re = imageClone(tt, bb, p, pixMax=100)
    ok, re, mask  = imageClone(tt, bb, p, mul=0.5, OSTU=True, Gauss=True)
    mm = 255 * np.ones(tt.shape, tt.dtype)
    # re = cv2.seamlessClone(tt, bb, mm, p, cv2.MIXED_CLONE)

    # output1 = cv2.seamlessClone(tt, bb, mm, (x,y), cv2.NORMAL_CLONE)
    # output2 = cv2.seamlessClone(tt, bb, mm, (x,y), cv2.MIXED_CLONE)
    # output3 = cv2.seamlessClone(tt, bb, mm, (x,y), cv2.MONOCHROME_TRANSFER)

    cv2.imshow(' ', re)
    cv2.waitKey(0)