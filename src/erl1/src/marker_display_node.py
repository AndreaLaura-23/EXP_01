
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32
from aruco_opencv_msgs.msg import ArucoDetection

from cv_bridge import CvBridge
import cv2
import numpy as np


class MarkerDisplayNode(Node):

    def __init__(self):
        super().__init__('marker_display_node')

        # Parametri ROS 2
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('detections_topic', '/aruco_detections')
        self.declare_parameter('output_topic', '/marker_image')
        self.declare_parameter('reached_topic', '/reached_marker_id')
        self.declare_parameter('marker_size', 0.2)
        self.declare_parameter('test_mode', True)

        def param(name):
            return self.get_parameter(name).value

        self.marker_size = param('marker_size')
        self.test_mode = param('test_mode')

        self.bridge = CvBridge()

        self.latest_image = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.detections = {}
        self.published_ids = set()

        self.create_subscription(
            Image, param('image_topic'),
            self.image_callback, 10
        )

        self.create_subscription(
            CameraInfo, param('camera_info_topic'),
            self.camera_info_callback, 10
        )

        self.create_subscription(
            ArucoDetection, param('detections_topic'),
            self.detections_callback, 10
        )

        self.create_subscription(
            Int32, param('reached_topic'),
            self.reached_callback, 10
        )

        self.image_pub = self.create_publisher(
            Image, param('output_topic'), 10
        )

        self.get_logger().info('Marker display avviato')

    def image_callback(self, msg):
        self.latest_image = msg

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(
            msg.k, dtype=np.float64
        ).reshape(3, 3)

        self.dist_coeffs = np.array(
            msg.d, dtype=np.float64
        )

    def detections_callback(self, msg):
        self.detections = {
            int(marker.marker_id): marker.pose
            for marker in msg.markers
        }

        if self.test_mode:
            for marker_id in self.detections:
                self.publish_marker(marker_id)

    def reached_callback(self, msg):
        if not self.test_mode:
            self.publish_marker(int(msg.data))

    def publish_marker(self, marker_id):

        if marker_id in self.published_ids:
            return

        if self.latest_image is None:
            return

        if self.camera_matrix is None:
            return

        if marker_id not in self.detections:
            return

        pose = self.detections[marker_id]

        # Trasformazione quaternione -> matrice di rotazione
        q = pose.orientation

        quaternion = np.array(
            [q.x, q.y, q.z, q.w],
            dtype=np.float64
        )

        norm = np.linalg.norm(quaternion)
        if norm < 1e-9:
            return

        x, y, z, w = quaternion / norm

        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ], dtype=np.float64)

        rvec, _ = cv2.Rodrigues(rotation)

        tvec = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z
        ], dtype=np.float64).reshape(3, 1)

        # Quattro angoli del marker in metri
        h = self.marker_size / 2.0

        object_points = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0]
        ], dtype=np.float32)

        # Proiezione 3D -> pixel
        points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        points = points.reshape(4, 2)

        center = np.mean(points, axis=0)
        cx, cy = center.astype(int)

        radius = int(
            max(np.linalg.norm(points - center, axis=1))
            + 10
        )

        frame = self.bridge.imgmsg_to_cv2(
            self.latest_image,
            desired_encoding='bgr8'
        ).copy()

        # Cerchio verde
        cv2.circle(
            frame, (cx, cy), radius,
            (0, 255, 0), 3
        )

        # ID del marker
        cv2.putText(
            frame,
            f'id {marker_id}',
            (max(0, cx-radius), max(25, cy-radius-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        output = self.bridge.cv2_to_imgmsg(
            frame, encoding='bgr8'
        )
        output.header = self.latest_image.header

        self.image_pub.publish(output)
        self.published_ids.add(marker_id)

        self.get_logger().info(
            f'Pubblicata immagine del marker {marker_id}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MarkerDisplayNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
