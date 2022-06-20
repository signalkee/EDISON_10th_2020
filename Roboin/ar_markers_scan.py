from __future__ import print_function

try:
    import cv2
    from ar_markers import detect_markers
except ImportError:
    raise Exception('Error: OpenCv is not installed')

def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)

    elif degrees == 180:
        dst = cv2.flip(src, -1)

    elif degrees == 270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    else:
        dst = null
    return dst

if __name__ == '__main__':
    print('Press "q" to quit')
    capture = cv2.VideoCapture(0)

    if capture.isOpened():  # try to get the first frame
        frame_captured, frame = capture.read()
    else:
        frame_captured = False
    while frame_captured:
        frame = Rotate(frame, 270)
        markers = detect_markers(frame)
        for marker in markers:
            marker.highlite_marker(frame)
        cv2.imshow('Test Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_captured, frame = capture.read()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
