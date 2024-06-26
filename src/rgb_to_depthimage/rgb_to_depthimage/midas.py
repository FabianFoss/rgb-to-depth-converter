import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch

from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from MiDaS.midas.model_loader import load_model

class DepthEstimator(Node):
    def __init__(self):
        super().__init__('midas_depth_estimate')
        self.publisher_ = self.create_publisher(Image, 'depth', 10)
        self.subscription = self.create_subscription(
            Image,
            'camera',
            self.listener_callback,
            10)
        self.bridge = CvBridge()

        # Choose the device you are working with
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Specify the model type and path to its weights
        self.model_type = "midas_v21_small_256"
        self.model_path = "/home/fabianfossbudal/repos/master-thesis-mono-repo/rgb-to-depth-converter/src/rgb_to_depth/resource/midas_v21_small_256.pt"

        # Load the model and the preprocessing transforms
        self.model, self.transform, _, _ = load_model(self.device, self.model_path, model_type=self.model_type, optimize=True)


    def listener_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert ROS Image message to OpenCV image: {str(e)}")
            return

        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing transforms and adjust the tensor's type and device
        input_tensor = torch.tensor(self.transform({"image": frame_rgb})["image"], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Perform inference to get the depth image
        with torch.no_grad():
            depth = self.model(input_tensor)

        # Convert depth map to a single-channel float32 image
        depth_float32 = depth.cpu().squeeze().numpy().astype(np.float32)

        # Normalize the depth values to approximate meters
        scaling_factor = 0.01  # Example scaling factor, adjust based on calibration
        depth_in_meters = depth_float32 * scaling_factor

        try:
            # Convert the depth image in meters to a ROS message with "32FC1" encoding
            depth_msg = self.bridge.cv2_to_imgmsg(depth_in_meters, "32FC1")
            
            # Copy the header from the input message to maintain the timestamp and frame_id
            depth_msg.header = msg.header
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(depth_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert depth image to ROS Image message: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    depth_estimator = DepthEstimator()
    rclpy.spin(depth_estimator)
    depth_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
