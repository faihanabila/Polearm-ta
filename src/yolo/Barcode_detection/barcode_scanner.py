import cv2
from pyzbar import pyzbar

class BarcodeDetector:
    def __init__(self):
        self.detected_barcodes = set()

    def process_frame(self, frame):
        """Process a video frame to detect and annotate barcodes/QR codes"""
        img = frame.copy()
        barcodes = pyzbar.decode(img)
        
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            barcode_text = f"{barcode_data} ({barcode_type})"
            
            if barcode_text not in self.detected_barcodes:
                self.detected_barcodes.add(barcode_text)
                print(f"Detected barcode: {barcode_text}")
                self._annotate_frame(img, barcode, barcode_text)    
        
        return img

    def _annotate_frame(self, img, barcode, text):
        """Draw visual annotations on detected barcodes"""
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def get_detected(self):
        """Return list of unique detected barcodes"""
        return list(self.detected_barcodes)

def main():
    # Pilih kamera (0 = default, 1 = webcam eksternal, bisa dicoba angka lain jika tidak terbaca)
    camera_index = 1  # Ubah ke 1 jika ingin menggunakan webcam eksternal
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"ERROR: Tidak dapat membuka kamera dengan indeks {camera_index}")
        return

    detector = BarcodeDetector()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Gagal mengambil frame dari kamera!")
                break

            # Proses frame dan tampilkan output
            annotated_frame = detector.process_frame(frame)
            cv2.imshow('Barcode Scanner', annotated_frame)

            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDetected codes:", detector.get_detected())

if __name__ == "__main__":
    main() 
    

