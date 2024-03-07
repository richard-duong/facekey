import cv2
import classifier as ci


def main():
    face_class = ci.face_class()
    
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()

        # Add detection box to video
        faces = ci.detect_box(frame, face_class) 

        # Display the resulting frame
        cv2.imshow("frame", frame)

        # the 'q' button is set as the
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
