import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera {index} gagal dibuka")
        return
    print(f"Camera {index} berhasil dibuka")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {index} gagal baca frame")
            break
        cv2.imshow(f"Camera {index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_camera(2)
    test_camera(4)
