import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')

# blank[300:400, 400:500] = 0, 255, 0
# blank[200:400, 300:500] = 0, 100, 200
cv.imshow('blank', blank)

cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=-1)
cv.imshow("Rectangle", blank)
 
cv.putText(blank, "Hello World", (220, 220), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 0), 3)
cv.imshow("text", blank)

if cv.waitKey(0) & 0xFF == ord('d'):
    cv.destroyAllWindows()