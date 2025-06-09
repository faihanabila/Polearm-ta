import cv2
import csv
import os
from datetime import datetime
from pyzbar import pyzbar
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


# Global set to store all barcodes saved to CSV (persist during runtime)
seen_barcodes_global = set()


# Dictionary to store barcode detection timestamps
barcode_detection_times = {}


class BarcodeDetector:
    def __init__(self, logger):
        """
        Initializes the BarcodeDetector with a logger.

        Args:
            logger: ROS2 logger object for logging messages.
        """
        self.logger = logger

    def process_frame(self, frame):
        """
        Processes a single video frame to detect and decode barcodes.
        Applies grayscale conversion and optional additional preprocessing.

        Args:
            frame: The input video frame (OpenCV BGR image).

        Returns:
            tuple: A tuple containing:
                - annotated_img: The frame with detected barcodes annotated.
                - new_detections: A list of (data, type) tuples for newly detected barcodes.
        """
        img = frame.copy()

        # --- Preprocessing Steps ---
        # 1. Convert to grayscale: Barcode detection often works better on grayscale images
        #    as it reduces complexity and focuses on intensity differences.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Optional: Apply Gaussian blur to reduce noise.
        #    Experiment with kernel size (e.g., (5, 5)) to find the best balance.
        #    Too much blur can obscure fine barcode lines.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Optional: Apply adaptive thresholding for better contrast, especially in uneven lighting.
        #    This converts the grayscale image to a binary (pure black and white) image.
        #    cv2.THRESH_BINARY + cv2.THRESH_OTSU automatically finds an optimal threshold.
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use the pre-processed image for barcode decoding.
        # You can choose 'gray', 'blurred', or 'thresh' based on your testing.
        # For this update, we'll use 'gray' as requested, but keep others commented for easy testing.
        barcodes = pyzbar.decode(gray)
        new_detections = []

        if not barcodes:
            # Display message if no barcodes are detected
            cv2.putText(img, "Tidak ada barcode terdeteksi", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            btype = barcode.type
            key = (data, btype)

            # Check if this barcode has been seen before globally
            if key not in seen_barcodes_global:
                text = f"{data} ({btype})"
                color = (0, 0, 255)  # Red for new detection
                new_detections.append((data, btype))
                self.logger.info(f"[NEW] {text}")
                barcode_detection_times[key] = time.time()  # Store the initial detection time
            else:
                # If seen before, check time difference for re-detection/logging purposes
                time_diff = time.time() - barcode_detection_times.get(key, 0)
                if time_diff > 5:  # If more than 5 seconds have passed since the last detection
                    text = "Barcode sudah terdeteksi (re-detected)"
                    color = (0, 255, 0)  # Green for re-detected
                    self.logger.info(f"[RE-DETECTED] {data}")
                    # Add to new_detections so it can be re-saved to CSV if needed
                    new_detections.append((data, btype))
                    # Update detection time to reset the 5-second window
                    barcode_detection_times[key] = time.time()
                else:
                    # If less than 5 seconds, it's considered recently seen, no new log/save
                    text = "Barcode sudah terdeteksi"
                    color = (0, 255, 0)  # Green for already seen
                    # No need to add to new_detections if it's within the cooldown period
                    self.logger.debug(f"[RECENTLY SEEN] {data}") # Use debug for less verbose logging

            # Annotate the original color frame for display
            self._annotate_frame(img, barcode, text, color)

        return img, new_detections

    def _annotate_frame(self, img, barcode, text, color):
        """
        Draws a rectangle and text around the detected barcode on the image.

        Args:
            img: The image frame to annotate.
            barcode: The decoded barcode object from pyzbar.
            text: The text to display (barcode data and type/status).
            color: The color for the rectangle and text (BGR tuple).
        """
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class WebcamBarcodeNode(Node):
    def __init__(self):
        """
        Initializes the ROS2 node for webcam barcode detection.
        Sets up camera capture, publishers, CSV logging, and barcode detector.
        """
        super().__init__('webcam_barcode_node')
        self.get_logger().info("Memulai node barcode webcam...")

        # Open video sources (adjust camera indices if needed)
        self.cap1 = cv2.VideoCapture(2)
        self.cap2 = cv2.VideoCapture(4)

        if not self.cap1.isOpened():
            self.get_logger().error("ERROR: Tidak dapat membuka Kamera 1 (index 2)!")
            # Attempt to open other cameras if index 2 fails
            self.cap1 = cv2.VideoCapture(0) # Try default camera
            if not self.cap1.isOpened():
                self.get_logger().error("ERROR: Tidak dapat membuka Kamera 1 (index 0)!")
                return # Exit if no camera can be opened

        if not self.cap2.isOpened():
            self.get_logger().error("ERROR: Tidak dapat membuka Kamera 2 (index 4)!")
            # Attempt to open other cameras if index 4 fails
            self.cap2 = cv2.VideoCapture(1) # Try another index
            if not self.cap2.isOpened():
                self.get_logger().error("ERROR: Tidak dapat membuka Kamera 2 (index 1)!")
                return # Exit if no camera can be opened


        self.detector = BarcodeDetector(self.get_logger())

        # ROS2 Publishers for barcode data
        self.publisher_1 = self.create_publisher(String, 'barcode_data_cam1', 10)
        self.publisher_2 = self.create_publisher(String, 'barcode_data_cam2', 10)

        # Setup CSV output directory and file name
        results_dir = '/home/faiha/polearm/src/barcode_detection/barcode_detection/results'
        os.makedirs(results_dir, exist_ok=True)

        today_str = datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
        self.csv_filename = os.path.join(results_dir, f"{today_str}.csv")

        # Open CSV file in append mode ('a')
        # Check if the file is new or empty to write headers only once
        write_header = not os.path.exists(self.csv_filename) or os.stat(self.csv_filename).st_size == 0
        self.csv_file = open(self.csv_filename, mode='a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)

        if write_header:
            self.csv_writer.writerow(['Timestamp', 'Data', 'Type', 'Camera', 'Time (seconds)'])
            self.get_logger().info(f"CSV file '{self.csv_filename}' created with header.")
        else:
            self.get_logger().info(f"Appending to existing CSV file '{self.csv_filename}'.")

        self.start_time = None # To track the total running time for detection

    def run(self):
        """
        Main loop for capturing frames, processing them, and publishing/saving barcode data.
        Handles user input for quitting or resetting barcode history.
        """
        while rclpy.ok():
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                self.get_logger().error("ERROR: Gagal membaca frame dari salah satu kamera!")
                # Attempt to re-open cameras if they fail
                if not self.cap1.isOpened(): self.cap1 = cv2.VideoCapture(2)
                if not self.cap2.isOpened(): self.cap2 = cv2.VideoCapture(4)
                time.sleep(1) # Wait a bit before trying again
                continue # Skip to next iteration

            # Start the timer when the first frame is successfully read
            if self.start_time is None:
                self.start_time = time.time()

            # Process frame from Camera 1
            annotated1, detected1 = self.detector.process_frame(frame1)
            cv2.imshow("Barcode Scanner Cam 1", annotated1)
            for data, btype in detected1:
                # Publish and save to CSV if it's a new detection or re-detection after cooldown
                # The logic in process_frame already handles the 5-second cooldown for 'new_detections'
                self.publisher_1.publish(String(data=data))
                self.save_to_csv(data, btype, "Cam1")

            # Process frame from Camera 2
            annotated2, detected2 = self.detector.process_frame(frame2)
            cv2.imshow("Barcode Scanner Cam 2", annotated2)
            for data, btype in detected2:
                # Publish and save to CSV if it's a new detection or re-detection after cooldown
                self.publisher_2.publish(String(data=data))
                self.save_to_csv(data, btype, "Cam2")

            # Handle key press for resetting barcode history (press 'r')
            key = cv2.waitKey(1) & 0xFF # Use & 0xFF for cross-platform compatibility
            if key == ord('r'):
                seen_barcodes_global.clear()
                barcode_detection_times.clear()  # Clear barcode detection times as well
                self.get_logger().info("Barcode history reset.")
            # Exit condition (press 'q' to quit)
            elif key == ord('q'):
                break

        self.cleanup()

    def save_to_csv(self, data, btype, camera):
        """
        Saves detected barcode data to the CSV file.
        Updates the global seen_barcodes_global set.

        Args:
            data (str): The decoded barcode data.
            btype (str): The type of barcode.
            camera (str): The camera from which the barcode was detected ("Cam1" or "Cam2").
        """
        # Calculate the time spent from the start of the application
        detection_time = round(time.time() - self.start_time, 2) if self.start_time is not None else 0.0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp format
        row = [timestamp, data, btype, camera, detection_time]
        self.csv_writer.writerow(row)
        self.csv_file.flush() # Ensure data is written to disk immediately

        # Add the barcode to the global set to mark it as seen
        seen_barcodes_global.add((data, btype))
        self.get_logger().info(f"[CSV] {row}")

    def cleanup(self):
        """
        Releases camera resources, closes OpenCV windows, and closes the CSV file.
        """
        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()
        self.csv_file.close()
        self.get_logger().info("Barcode scanner ditutup.")


def main(args=None):
    """
    Main function to initialize and run the ROS2 barcode detection node.
    Ensures proper shutdown of ROS2 and resources.
    """
    rclpy.init(args=args)
    node = WebcamBarcodeNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user (Ctrl+C). Shutting down.")
    finally:
        # Ensure cleanup and shutdown are called even if an error occurs
        node.cleanup()
        rclpy.shutdown()


if __name__ == '__main__':
    main()