import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from pyzbar import pyzbar


class BarcodeDetector:
    def __init__(self):
        self.detected_barcodes = []

    def process_frame(self, frame):
        img = frame.copy()
        barcodes = pyzbar.decode(img)

        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            barcode_text = f"{barcode_data} ({barcode_type})"

            # Tampilkan bounding box setiap frame
            self._annotate_frame(img, barcode, barcode_text)

            # Simpan data barcode sebagai tuple (barcode_data, barcode_type)
            if (barcode_data, barcode_type) not in self.detected_barcodes:
                self.detected_barcodes.append((barcode_data, barcode_type))  # Gunakan tuple

        return img

    def _annotate_frame(self, img, barcode, text):
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def get_detected(self):
        return self.detected_barcodes  # Mengembalikan list of tuples (data, type)


class WebcamBarcodeNode(Node):
    def __init__(self):
        super().__init__('webcam_barcode_node')
        self.get_logger().info("Starting webcam barcode node...")

        # Mendeklarasikan publisher untuk mengirimkan data barcode
        self.publisher_ = self.create_publisher(String, 'barcode_data', 10)
        self.detector = BarcodeDetector()
        self.cap = cv2.VideoCapture(2)  # Coba dengan /dev/video2 atau index yang sesuai

        if not self.cap.isOpened():
            self.get_logger().error("ERROR: Tidak dapat membuka kamera!")
            return

        self.run()

    def run(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("ERROR: Gagal membaca frame dari kamera!")
                break

            annotated = self.detector.process_frame(frame)
            cv2.imshow("Barcode Scanner", annotated)

            # Publikasikan data barcode yang terdeteksi
            detected_barcodes = self.detector.get_detected()
            for data, btype in detected_barcodes:
                self.get_logger().info(f"📦 Terdeteksi: {data} ({btype})")
                # Publikasikan data barcode ke topik ROS
                self.publisher_.publish(String(data=data))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("Barcode scanner ditutup.")
        self.get_logger().info(f"Detected barcodes: {self.detector.get_detected()}")


def main(args=None):
    rclpy.init(args=args)
    node = WebcamBarcodeNode()
    node.run()  # Jalankan proses pengambilan gambar dan pemrosesan barcode
    rclpy.shutdown()


if __name__ == '__main__':
    main()