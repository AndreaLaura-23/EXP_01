#!/usr/bin/env python3

import cv2
import math
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from aruco_opencv_msgs.msg import ArucoDetection


class MarkerImageNode(Node):

    def __init__(self):
        super().__init__('marker_image_node')

        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('detections_topic', '/aruco_detections')
        self.declare_parameter('output_topic', '/marker_image')
        self.declare_parameter('reached_topic', '/marker_reached')
        self.declare_parameter('marker_size', 0.2)

        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        output_topic = self.get_parameter('output_topic').value
        reached_topic = self.get_parameter('reached_topic').value

        self.marker_size = float(
            self.get_parameter('marker_size').value
        )

        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_detections = None
        self.camera_info = None

        self.previous_reached = False

        self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )

        self.create_subscription(
            ArucoDetection,
            detections_topic,
            self.detections_callback,
            10
        )

        self.create_subscription(
            Bool,
            reached_topic,
            self.reached_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            output_topic,
            10
        )

        self.get_logger().info(
            'Marker image node started'
        )

    def image_callback(self, msg):
        self.latest_image = msg

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def detections_callback(self, msg):
        self.latest_detections = msg

    def reached_callback(self, msg):

        # Pubblica soltanto sul passaggio False -> True.
        rising_edge = msg.data and not self.previous_reached
        self.previous_reached = msg.data

        if not rising_edge:
            return

        if self.latest_image is None:
            self.get_logger().warn('Camera image not available')
            return

        if self.camera_info is None:
            self.get_logger().warn('Camera info not available')
            return

        if self.latest_detections is None:
            self.get_logger().warn('ArUco detections not available')
            return

        markers = self.latest_detections.markers

        if not markers:
            self.get_logger().warn('No ArUco marker detected')
            return

        # Verifica che la posa sia nel frame ottico della camera.
        if (
            self.latest_detections.header.frame_id
            != self.camera_info.header.frame_id
        ):
            self.get_logger().warn(
                'Detection and camera_info frames differ: '
                f'{self.latest_detections.header.frame_id} vs '
                f'{self.camera_info.header.frame_id}'
            )
            return

        # Evita di usare un'immagine e una rilevazione
        # troppo lontane nel tempo.
        image_stamp = self.latest_image.header.stamp
        detection_stamp = self.latest_detections.header.stamp

        image_time = image_stamp.sec + image_stamp.nanosec * 1e-9
        detection_time = (
            detection_stamp.sec + detection_stamp.nanosec * 1e-9
        )

        if abs(image_time - detection_time) > 0.5:
            self.get_logger().warn(
                'Image and detections are not synchronized'
            )
            return

        image = self.bridge.imgmsg_to_cv2(
            self.latest_image,
            desired_encoding='bgr8'
        ).copy()

        height, width = image.shape[:2]

        # Matrice intrinseca della camera.
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        if fx <= 0 or fy <= 0:
            self.get_logger().warn('Invalid camera calibration')
            return

        # Se più marker sono visibili, scegli quello
        # il cui centro è più vicino al centro dell'immagine.
        candidates = []

        for marker in markers:

            x = marker.pose.position.x
            y = marker.pose.position.y
            z = marker.pose.position.z

            if z <= 0:
                continue

            u = fx * x / z + cx
            v = fy * y / z + cy

            if not (0 <= u < width and 0 <= v < height):
                continue

            distance = math.hypot(
                u - width / 2,
                v - height / 2
            )

            candidates.append(
                (distance, marker, u, v, z)
            )

        if not candidates:
            self.get_logger().warn(
                'No marker center projects inside the image'
            )
            return

        _, marker, u, v, z = min(
            candidates,
            key=lambda item: item[0]
        )

        # Raggio indicativo basato sulla dimensione
        # apparente del marker.
        apparent_side = (
            (fx + fy) / 2.0 * self.marker_size / z
        )

        radius = max(
            10,
            int(apparent_side * 0.75)
        )

        center = (int(round(u)), int(round(v)))

        cv2.circle(
            image,
            center,
            radius,
            (0, 255, 0),
            3
        )

        label = f'ID: {marker.marker_id}'

        text_x = max(0, center[0] - radius)
        text_y = max(25, center[1] - radius - 10)

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        output_msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )

        output_msg.header = self.latest_image.header

        self.publisher.publish(output_msg)

        self.get_logger().info(
            f'Published marker image with ID {marker.marker_id}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = MarkerImageNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
