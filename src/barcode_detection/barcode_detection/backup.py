#!/usr/bin/env python3
import cv2
import csv
import os
import subprocess
import time
from datetime import datetime
from pyzbar import pyzbar
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

seen_barcodes_global = set()
barcode_detection_times = {}

class BarcodeDetector:
    def __init__(self, logger):
        self.logger = logger

    def process_frame(self, frame):
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        new_detections = []

        if not barcodes:
            cv2.putText(img, "No barcode detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            btype = barcode.type
            key = (data, btype)

            if key not in seen_barcodes_global:
                text = f"{data} ({btype})"
                color = (0, 0, 255)  # Red for new detection
                new_detections.append((data, btype))
                self.logger.info(f"[NEW] {text}")
                barcode_detection_times[key] = time.time()
            else:
                time_diff = time.time() - barcode_detection_times.get(key, 0)
                if time_diff > 5:  # 5 seconds threshold for re-detection
                    text = "Barcode re-detected"
                    color = (0, 255, 0)  # Green for re-detection
                    new_detections.append((data, btype))
                    self.logger.info(f"[RE-DETECTED] {data}")
                    barcode_detection_times[key] = time.time()
                else:
                    text = "Barcode already detected"
                    color = (0, 255, 0)  # Green for already detected
                    self.logger.debug(f"[SEEN] {data}")

            self._annotate_frame(img, barcode, text, color)

        return img, new_detections

    def _annotate_frame(self, img, barcode, text, color):
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

class WebcamBarcodeNode(Node):
    def __init__(self):
        super().__init__('webcam_barcode_node')
        self.get_logger().info("Starting barcode detection node...")

        # Deklarasi parameter
        self.declare_parameter('camera1_device', '/dev/video2')
        self.declare_parameter('camera2_device', '/dev/video6')
        self.declare_parameter('use_v4l2', True)
        self.declare_parameter('show_camera', True)
        self.declare_parameter('results_dir', '/home/faiha/Polearm/src/barcode_detection/results')

        # Dapatkan parameter
        cam1 = self.get_parameter('camera1_device').value
        cam2 = self.get_parameter('camera2_device').value
        use_v4l2 = self.get_parameter('use_v4l2').value
        self.show_camera = self.get_parameter('show_camera').value
        results_dir = self.get_parameter('results_dir').value

        # Buka kamera
        if use_v4l2:
            self.cap1 = cv2.VideoCapture(cam1, cv2.CAP_V4L2)
            self.cap2 = cv2.VideoCapture(cam2, cv2.CAP_V4L2)
        else:
            self.cap1 = cv2.VideoCapture(cam1)
            self.cap2 = cv2.VideoCapture(cam2)

        # Verifikasi kamera
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            self.get_logger().error("Failed to open one or both cameras!")
            raise RuntimeError("Camera initialization failed")

        self.get_logger().info(f"Successfully opened cameras: {cam1} and {cam2}")

        # Inisialisasi publisher
        self.publisher_1 = self.create_publisher(String, 'barcode_data_cam1', 10)
        self.publisher_2 = self.create_publisher(String, 'barcode_data_cam2', 10)

        # Setup CSV logging
        os.makedirs(results_dir, exist_ok=True)
        today_str = datetime.now().strftime("%Y%m%d")
        self.csv_filename = os.path.join(results_dir, f"{today_str}.csv")
        
        write_header = not os.path.exists(self.csv_filename) or os.stat(self.csv_filename).st_size == 0
        self.csv_file = open(self.csv_filename, mode='a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        if write_header:
            self.csv_writer.writerow(['Timestamp', 'Data', 'Type', 'Camera', 'Time (seconds)'])
        
        self.start_time = time.time()
        self.detector = BarcodeDetector(self.get_logger())

    def run(self):
        while rclpy.ok():
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                self.get_logger().error("Failed to read frame from one or both cameras")
                time.sleep(0.1)
                continue

            # Process camera 1
            annotated1, detected1 = self.detector.process_frame(frame1)
            for data, btype in detected1:
                self.publisher_1.publish(String(data=data))
                self._save_to_csv(data, btype, "Cam1")
            
            # Process camera 2
            annotated2, detected2 = self.detector.process_frame(frame2)
            for data, btype in detected2:
                self.publisher_2.publish(String(data=data))
                self._save_to_csv(data, btype, "Cam2")

            # Display frames if enabled
            if self.show_camera:
                cv2.imshow("Camera 1", annotated1)
                cv2.imshow("Camera 2", annotated2)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    seen_barcodes_global.clear()
                    barcode_detection_times.clear()
                    self.get_logger().info("Reset barcode history")

        self.cleanup()

    def _save_to_csv(self, data, btype, camera):
        detection_time = round(time.time() - self.start_time, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, data, btype, camera, detection_time]
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        seen_barcodes_global.add((data, btype))
        self.get_logger().info(f"Saved to CSV: {data}")

    def cleanup(self):
        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()
        self.csv_file.close()
        self.get_logger().info("Barcode scanner closed.")

def main(args=None):
    rclpy.init(args=args)
    node = WebcamBarcodeNode()
    
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user")
    except Exception as e:
        node.get_logger().error(f"Error: {str(e)}")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()