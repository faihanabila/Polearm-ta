import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from pyzbar import pyzbar


class BarcodeDetector:
    def __init__(self):
        self.detected_barcodes = set()

    def process_frame(self, frame):
        img = frame.copy()
        barcodes = pyzbar.decode(img)

        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            barcode_text = f"{barcode_data} ({barcode_type})"

            # Tampilkan bounding box setiap frame
            self._annotate_frame(img, barcode, barcode_text)

            if barcode_text not in self.detected_barcodes:
                self.detected_barcodes.add(barcode_text)
                print(f"[DETECTED] {barcode_text}")

        return img

    def _annotate_frame(self, img, barcode, text):
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def get_detected(self):
        return list(self.detected_barcodes)


class WebcamBarcodeNode(Node):
    def __init__(self):
        super().__init__('webcam_barcode_node')
        self.get_logger().info("Starting webcam barcode node...")

        # Ambil parameter cam_index dari ROS (default /dev/video2)
        self.declare_parameter('cam_index', 2)  # Default ke 2 jika tidak diatur
        cam_index = self.get_parameter('cam_index').get_parameter_value().integer_value
        self.get_logger().info(f"🎯 Membuka kamera dengan index {cam_index}...")

        # Buka kamera menggunakan cv2.VideoCapture
        self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            self.get_logger().error(f"ERROR: Tidak dapat membuka kamera dengan index {cam_index}!")
            return

        self.publisher_ = self.create_publisher(String, 'barcode_data', 10)
        self.detector = BarcodeDetector()

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("ERROR: Gagal membaca frame dari kamera!")
            return

        annotated = self.detector.process_frame(frame)
        cv2.imshow("Barcode Scanner", annotated)

        for data, btype in self.detector.detected_barcodes:
            self.publisher_.publish(String(data=data))
            self.get_logger().info(f"📦 Terdeteksi: {data} ({btype})")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def run(self):
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)
    node = WebcamBarcodeNode()
    node.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()