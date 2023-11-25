import cv2
import numpy as np
import random

def gaussian2D(shape, sigma=0.02):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h



def augGaus(img):
    img = np.float32(img)/255
    h, w = img.shape
    t = random.randint(0, 3)
    # t = 1

    for i in range(t):
        # size1 = random.randint(int(w / 2), int(h))
        size1 = 2*max(w, h)
        size2 = size1 * random.uniform(0.66, 1.5)
        sigma = random.randint(100, 400)
        randomGR = random.uniform(0.1, 0.5)
        # randomGR = random.uniform(0.8, 0.9)
        # sigma = random.randint(800, 2000)
        auxImg = gaussian2D((size1, size2), sigma=sigma)
        # auxImg = cv2.normalize(auxImg, None, 0, 1, cv2.NORM_MINMAX)
        # cv2.imshow(' ', auxImg)
        # cv2.waitKey(0)
        xCenter = random.randint(0, w-1)
        yCenter = random.randint(0, h-1)

        # placeXLeft = placeX -

        # h, w = dst.shape
        m, n = auxImg.shape
        xLeft, yTop = xCenter - (n - 1) // 2, yCenter - (m - 1) // 2
        xRight, yDown = xLeft + n, yTop + m
        srcLeft, srcRight, srcTop, srcDown = 0, n, 0, m
        if xLeft < 0:
            srcLeft = -xLeft
            xLeft = 0
        if xRight > w:
            srcRight = -xRight + w
            xRight = w
        if yTop < 0:
            srcTop = -yTop
            yTop = 0
        if yDown > h:
            srcDown = -yDown + h
            yDown = h

        # a123 = cv2.normalize(auxImg[srcTop:srcDown, srcLeft:srcRight], None, 0, 1, cv2.NORM_MINMAX)
        a123 = 1 - auxImg[srcTop:srcDown, srcLeft:srcRight]*randomGR
        if random.random() > 0.5:
            img[yTop:yDown, xLeft:xRight] = img[yTop:yDown, xLeft:xRight] * a123
        else:
            img[yTop:yDown, xLeft:xRight] = img[yTop:yDown, xLeft:xRight] / a123
        # img = np.clip(img*255, 0, 255)
        # img = np.uint8(img)

        # cv2.imshow(' ', a123)
        # cv2.waitKey(0)
        # img[yTop:yDown, xLeft:xRight] = img[yTop:yDown, xLeft:xRight] / (1-auxImg[srcTop:srcDown, srcLeft:srcRight])

    img = np.clip(img * 255, 0, 255)
    img = np.uint8(img)
    # cv2.imshow(' ', img)
    # cv2.waitKey(0)
    return img

if __name__ == '__main__':
    h, w = 600, 800
    # img = np.ones((h, w), dtype=np.float32)/2
    img = np.ones((h, w), dtype=np.uint8)*117
    augGaus(img)

# if __name__ == '__main__':
#     h, w = 600, 800
#     img = np.ones((h, w), dtype=np.float32)/2
#     t = random.randint(0, 9)
#
#     for i in range(t):
#         # size1 = random.randint(int(w / 2), int(h))
#         size1 = 8000
#         size2 = size1 * random.uniform(0.66, 1.5)
#         sigma = random.randint(100, 200)
#         # sigma = random.randint(800, 2000)
#         auxImg = gaussian2D((size1, size2), sigma=sigma)
#         # auxImg = cv2.normalize(auxImg, None, 0, 1, cv2.NORM_MINMAX)
#         # cv2.imshow(' ', auxImg)
#         # cv2.waitKey(0)
#         xCenter = random.randint(0, w-1)
#         yCenter = random.randint(0, h-1)
#
#         # placeXLeft = placeX -
#
#         # h, w = dst.shape
#         m, n = auxImg.shape
#         xLeft, yTop = xCenter - (n - 1) // 2, yCenter - (m - 1) // 2
#         xRight, yDown = xLeft + n, yTop + m
#         srcLeft, srcRight, srcTop, srcDown = 0, n, 0, m
#         if xLeft < 0:
#             srcLeft = -xLeft
#             xLeft = 0
#         if xRight > w:
#             srcRight = -xRight + w
#             xRight = w
#         if yTop < 0:
#             srcTop = -yTop
#             yTop = 0
#         if yDown > h:
#             srcDown = -yDown + h
#             yDown = h
#
#         # a123 = cv2.normalize(auxImg[srcTop:srcDown, srcLeft:srcRight], None, 0, 1, cv2.NORM_MINMAX)
#         a123 = auxImg[srcTop:srcDown, srcLeft:srcRight]*0.2
#         img[yTop:yDown, xLeft:xRight] = img[yTop:yDown, xLeft:xRight] / (1-a123)
#         img = np.clip(img, 0, 1.0)
#         # cv2.imshow(' ', a123)
#         # cv2.waitKey(0)
#         # img[yTop:yDown, xLeft:xRight] = img[yTop:yDown, xLeft:xRight] / (1-auxImg[srcTop:srcDown, srcLeft:srcRight])
#
#         cv2.imshow(' ', img)
#         cv2.waitKey(0)