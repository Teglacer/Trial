#试一试
import cv2
import numpy as np

name = input('输入视频名：')
output = input('输入输出视频地址名：')
video = cv2.VideoCapture(name)
# 判断视频是否打开
if video.isOpened():
    print('视频开始处理，请稍后')
else:
    print('视频无法打开')

# 测试用,查看视频size 'D:\OpenCVcourse\vtest.avi' D:\OpenCVcourse\extraction.avi
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(output, fourcc, video.get(cv2.CAP_PROP_FPS), size)

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
kernel = np.ones((5, 5), np.uint8)
background = None

while video.isOpened():
    # 读取视频流
    grabbed, frame_lwpCV = video.read()
    if frame_lwpCV is None:
        break
    # 对帧进行预处理，先转灰度图，再进行高斯滤波。
    # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 将第一帧设置为整个输入的背景
    if background is None:
        background = gray_lwpCV
        continue
    # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
    # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
    diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀

    # 显示矩形框
    _,contours, hierarchy = cv2.findContours(diff.copy(), mode=cv2.RETR_EXTERNAL,method= cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        if cv2.contourArea(c) < 2500:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
        count += 1  # 检测到运动目标帧，计数器+1
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #print(count)
    if count > 1:
        writer.write(frame_lwpCV)

    background = gray_lwpCV

    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
# When everything done, release the capture
video.release()
writer.release()
print('视频处理完毕')
cv2.destroyAllWindows()
