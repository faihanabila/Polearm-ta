import cv2
import csv
import os
import time
from datetime import datetime
from pyzbar import pyzbar
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import atexit

seen_barcodes_global = set()

class BarcodeDetector:
    def __init__(self, logger):
        self.logger = logger

    def process_frame(self, frame):
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        new_detections = []

        if not barcodes:
            cv2.putText(img, "Tidak ada barcode terdeteksi", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            btype = barcode.type
            key = (data, btype)

            if key not in seen_barcodes_global:
                text = f"{data} ({btype})"
                color = (0, 0, 255)
                new_detections.append((data, btype))  # hanya jika belum pernah dilihat
                self.logger.info(f"[NEW] {text}")
            else:
                text = f"{data} [SUDAH TERDETEKSI]"
                color = (0, 255, 0)

            self._annotate_frame(img, barcode, text, color)

        return img, new_detections

    def _annotate_frame(self, img, barcode, text, color):
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class WebcamBarcodeNode(Node):
    def __init__(self):
        super().__init__('webcam_barcode_node')
        self.get_logger().info("Memulai node barcode webcam...")

        # Kamera dan path
        self.declare_parameter('camera1_device', '/dev/video2')
        self.declare_parameter('camera2_device', '/dev/video4')
        self.declare_parameter('use_v4l2', False)
        self.declare_parameter('results_dir', '/home/faiha/Polearm/src/barcode_detection/results')

        cam1 = self.get_parameter('camera1_device').value
        cam2 = self.get_parameter('camera2_device').value
        use_v4l2 = self.get_parameter('use_v4l2').value
        self.results_dir = self.get_parameter('results_dir').value

        if use_v4l2:
            self.cap1 = cv2.VideoCapture(cam1, cv2.CAP_V4L2)
            self.cap2 = cv2.VideoCapture(cam2, cv2.CAP_V4L2)
        else:
            self.cap1 = cv2.VideoCapture(cam1)
            self.cap2 = cv2.VideoCapture(cam2)

        time.sleep(1)

        if not self.cap1.isOpened():
            self.get_logger().error(f"❌ Kamera 1 ({cam1}) gagal dibuka.")
        if not self.cap2.isOpened():
            self.get_logger().error(f"❌ Kamera 2 ({cam2}) gagal dibuka.")
        if not self.cap1.isOpened() or not self.cap2.isOpened():
             return

        self.detector = BarcodeDetector(self.get_logger())
        self.publisher_1 = self.create_publisher(String, 'barcode_data_cam1', 10)
        self.publisher_2 = self.create_publisher(String, 'barcode_data_cam2', 10)

        os.makedirs(self.results_dir, exist_ok=True)
        today_str = datetime.now().strftime("%Y%m%d")
        self.csv_filename = os.path.join(self.results_dir, f"{today_str}.csv")
        write_header = not os.path.exists(self.csv_filename) or os.stat(self.csv_filename).st_size == 0
        self.csv_file = open(self.csv_filename, mode='a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)

        if write_header:
            self.csv_writer.writerow(['Timestamp', 'Data', 'Type', 'Camera', 'Time (seconds)'])
            self.get_logger().info(f"[CSV HEADER] File dibuat: {self.csv_filename}")
        else:
            self.get_logger().info(f"[CSV EXISTING] Menambahkan ke: {self.csv_filename}")

        self.start_time = time.time()
        atexit.register(self.cleanup)

    def run(self):
        while rclpy.ok():
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                self.get_logger().error("ERROR: Gagal membaca frame.")
                time.sleep(1)
                continue

            annotated1, detected1 = self.detector.process_frame(frame1)
            cv2.imshow("Barcode Scanner Cam 1", annotated1)
            for data, btype in detected1:
                key = (data, btype)
                if key not in seen_barcodes_global:
                    self.publisher_1.publish(String(data=data))
                    self.save_to_csv(data, btype, "Cam1")
                    seen_barcodes_global.add(key)

            annotated2, detected2 = self.detector.process_frame(frame2)
            cv2.imshow("Barcode Scanner Cam 2", annotated2)
            for data, btype in detected2:
                key = (data, btype)
                if key not in seen_barcodes_global:
                    self.publisher_2.publish(String(data=data))
                    self.save_to_csv(data, btype, "Cam2")
                    seen_barcodes_global.add(key)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                seen_barcodes_global.clear()
                self.get_logger().info("Barcode history di-reset.")
            elif key == ord('q'):
                break

        self.cleanup()

    def save_to_csv(self, data, btype, camera):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detection_time = int(time.time() - self.start_time)  # 🔁 Detik bulat
        row = [timestamp, data, btype, camera, detection_time]
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())
        self.get_logger().info(f"[CSV] {row}")

    def cleanup(self):
        if hasattr(self, 'cap1') and self.cap1:
            self.cap1.release()
        if hasattr(self, 'cap2') and self.cap2:
            self.cap2.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
            self.get_logger().info(f"[CLOSE FILE] {self.csv_filename}")
        self.get_logger().info("Scanner dimatikan.")

def main(args=None):
    rclpy.init(args=args)
    node = WebcamBarcodeNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Berhenti (Ctrl+C).")
    finally:
        node.cleanup()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
