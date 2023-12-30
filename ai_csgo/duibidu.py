import cv2
import numpy as np
def enhance_contrast_and_brightness(image_path, alpha=1.5, beta=50):
    # 读取图像
    img = cv2.imread(image_path)

    # 使用alpha调整对比度，beta调整亮度
    enhanced_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 显示结果
    cv2.imshow('原图片', img)
    cv2.imshow('结果', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用函数并传入图像路径
#enhance_contrast_and_brightness('path/to/your/image.jpg', alpha=1.5, beta=50)
# 用法示例
image_path = 'C:/Users/12051/Desktop/ai_csgo/abc.png'
enhance_contrast_and_brightness(image_path, alpha=1.5, beta=50)