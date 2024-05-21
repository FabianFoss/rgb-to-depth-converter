import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSHistoryPolicy
import torch
import cv2
import numpy as np
from DistDepth.networks.depth_decoder import DepthDecoder
from DistDepth.networks.resnet_encoder import ResnetEncoder
from DistDepth.utils import output_to_depth
import time
from sensor_msgs.msg import CameraInfo

class DepthEstimator(Node):
    def __init__(self):
        super().__init__('distdepth_depth_estimate')
        qos_profile = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST)

        self.image_subscribe = self.create_subscription(
            Image,
            '/camera',
            self.image_callback,
            qos_profile
        )
        pub_qos_profile = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST)
        
        self.depth_image_publisher = self.create_publisher(Image, '/depth', pub_qos_profile)        
        
        self.depth_camera_info_publisher = self.create_publisher(CameraInfo, 'depth_camera_info', 10)
        self.timer = self.create_timer(1.0, self.publish_camera_info)
        
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder, self.depth_decoder = self.load_model()

    def load_model(self):
        encoder = ResnetEncoder(152, False)
        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

        # Load pretrained models
        encoder_dict = torch.load('/home/fabianfossbudal/repos/master-thesis-mono-repo/rgb-to-depth-converter/src/rgb_to_depthimage/resource/ckpts/encoder.pth', map_location=self.device)
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        encoder.to(self.device)
        encoder.eval()

        depth_dict = torch.load('/home/fabianfossbudal/repos/master-thesis-mono-repo/rgb-to-depth-converter/src/rgb_to_depthimage/resource/ckpts/depth.pth', map_location=self.device)
        depth_decoder.load_state_dict(depth_dict)
        depth_decoder.to(self.device)
        depth_decoder.eval()

        return encoder, depth_decoder

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            input_tensor = torch.from_numpy(np.transpose(cv_image, (2, 0, 1))).float().to(self.device)
            input_tensor = (input_tensor / 255.0).unsqueeze(0)
            input_tensor = torch.nn.functional.interpolate(input_tensor, (256, 256), mode='bilinear', align_corners=False)

            with torch.no_grad():
                features = self.encoder(input_tensor)
                outputs = self.depth_decoder(features)
                depth_output = outputs[("out", 0)]  # Replace this with the correct key as per your model's output
                depth = output_to_depth(depth_output, 0.1, 10)
            
            # Keep the depth data as floating point
            depth_image_float = depth.cpu().numpy().squeeze() * 0.5 # Prototype adjustments

            # Create a depth image message using the correct float encoding
            depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image_float, encoding="32FC1")
            depth_image_msg.header = msg.header
            self.depth_image_publisher.publish(depth_image_msg)

        except Exception as e:
            self.get_logger().error('Failed to convert and publish depth image: {}'.format(e))

   
    def publish_camera_info(self):
        camera_info = get_synthetic_camera_info(256, 256, 22)
        camera_info.header.stamp = self.get_clock().now().to_msg()
        self.depth_camera_info_publisher.publish(camera_info)

def get_synthetic_camera_info(width, height, fov_degrees):

    fov = fov_degrees * (np.pi / 180)

    focal_length = width / (2.0 * np.tan(fov / 2))

    camera_info = CameraInfo()
    camera_info.header.frame_id = "front_camera_depth"
    camera_info.width = width
    camera_info.height = height
    camera_info.k = [focal_length, 0.0, width / 2,
                     0.0, focal_length, height / 2,
                     0.0, 0.0, 1.0]
    camera_info.p = [focal_length, 0.0, width / 2, 0.0,
                     0.0, focal_length, height / 2, 0.0,
                     0.0, 0.0, 1.0, 0.0]
    camera_info.r = [1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0]
    camera_info.distortion_model = 'plumb_bob'
    camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Assuming no distortion

    return camera_info


def main(args=None):
    rclpy.init(args=args)
    depth_estimator = DepthEstimator()
    rclpy.spin(depth_estimator)
    depth_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
