import cv2
import os

##### VIDEO CAPTURE EXAMPLE #####

cap = cv2.VideoCapture(1)

numFrame = 0
# Need to create the folder first
path = 'dataSet/images/stop/'

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bwframe = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY) # SOGLIA CIRCA 220

    # Saves the current frame
    numFrame += 1
    cv2.imwrite(os.path.join(path, '%d.jpg' % numFrame), bwframe)

    # Display the resulting frame
    cv2.imshow('frame', bwframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()