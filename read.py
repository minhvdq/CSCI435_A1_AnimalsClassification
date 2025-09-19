import cv2 as cv

img = cv.imread('Animals10/Bird/1.jpg')
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    cv.imshow('hi', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
def resize_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    # return frame

# img_resized = resize_frame(img, 1.5)
# cv.imshow('Image', img)
# print(img.shape[0], " height")
# img_resized = resize_frame(img, 10)
# cv.imshow('Image Resized', img_resized)
cv.waitKey(0)
cv.destroyAllWindows()