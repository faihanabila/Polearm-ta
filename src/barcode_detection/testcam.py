import cv2

# Inisialisasi video capture dari kamera pertama
cap = cv2.VideoCapture(0)

# Set format ke MJPEG (RGB)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Tentukan resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Cek frame kamera
ret, frame = cap.read()
if ret:
    cv2.imshow("Camera", frame)

# Tunggu input dari pengguna sebelum menutup jendela
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
