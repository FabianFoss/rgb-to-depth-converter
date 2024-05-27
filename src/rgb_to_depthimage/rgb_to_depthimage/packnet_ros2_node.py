import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import torch
import cv2

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank
from packnet_sfm.utils.image import interpolate_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.depth import inv2depth

from packnet_sfm.utils.types import is_seq, is_tensor

"""
NOTE: This requires cuda

NOTE: 
This code tries to convert this: https://github.com/surfii3z/packnet_sfm_ros/blob/master/ros/packnet_sfm_node 
From ros1 to ros2

NOTE: This code is partially generated using ChatGPT
"""

STEREO_SCALE_FACTOR = 1
MODEL_NAME = "PackNet01_HR_velsup_CStoK.ckpt"
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

class PacknedDepthEstimate(Node):
    def __init__(self):
        super().__init__('packnet_depth_estimate')
        self.bridge = CvBridge()
        self.model_wrapper = None
        self.network_input_shape = None
        self.original_input_shape = None

        self.pub_rgb_image = self.create_publisher(Image, '/packnet/color/image_raw', 1)
        self.pub_depth_image = self.create_publisher(Image, '/packnet/depth/image_raw', 1)
        self.subscriber = self.create_subscription(Image, 'camera', self.cb_image, 1)

        self.set_model_wrapper()

    def set_model_wrapper(self):
        
        models_path = 'src/packnet_sfm_ros/trained_models/'
        models_name = MODEL_NAME
        config, state_dict = parse_test_file("~/repos/master-thesis-mono-repo/rgb-to-depth-converter/src/rgb_to_depth/resource/PackNet01_MR_velsup_CStoK.ckpt")
        
        self.set_network_input_shape(config)

        # Initialize model wrapper from checkpoint arguments
        self.model_wrapper = ModelWrapper(config, load_datasets=False)
        # Restore monodepth_model state
        self.model_wrapper.load_state_dict(state_dict)

        if torch.cuda.is_available():
            self.model_wrapper = self.model_wrapper.to('cuda:{}'.format(rank()), dtype=None)
        
        # Set to eval mode
        self.model_wrapper.eval()

    def set_network_input_shape(self, config):
        self.network_input_shape = config['datasets']['augmentation']['image_shape']

    def process(self, rgb_img_msg):
        starter.record()
        self.get_logger().info("process seq: {}".format(rgb_img_msg.header.seq))
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            
        # shrink the image to fit NN input
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (self.network_input_shape[1], self.network_input_shape[0]), interpolation=cv2.INTER_LANCZOS4)
        rgb_image = to_tensor(rgb_image).unsqueeze(0)

        if torch.cuda.is_available():
            rgb_image = rgb_image.to('cuda:{}'.format(rank()), dtype=None)
        
        # Depth inference (returns predicted inverse depth)
        pred_inv_depth = self.model_wrapper.depth(rgb_image)

        # resize from PIL image and cv2 has different convention about the image shape 
        pred_inv_depth_resized = interpolate_image(pred_inv_depth, (self.original_input_shape[0], self.original_input_shape[1]), mode='bilinear', align_corners=False)
        
        # convert inverse depth to depth image
        depth_img = self.write_depth(self.inv2depth(pred_inv_depth_resized))

        depth_img_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="mono16")

        # define the header
        rgb_img_msg.header.stamp = self.get_clock().now()
        depth_img_msg.header.stamp = self.get_clock().now()
        rgb_img_msg.header.frame_id = "camera_depth_frame"
        depth_img_msg.header.frame_id = "camera_depth_frame"
        depth_img_msg.header.seq = rgb_img_msg.header.seq
        
        # publish the image and depth_image
        self.pub_rgb_image.publish(rgb_img_msg)
        self.pub_depth_image.publish(depth_img_msg)

        ender.record()

        torch.cuda.synchronize()
        process_time = starter.elapsed_time(ender)
        self.get_logger().info("process time: {}".format(process_time))


    def inv2depth(self, inv_depth):
        """
        Invert an inverse depth map to produce a depth map

        Parameters
        ----------
        inv_depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
            Inverse depth map

        Returns
        -------
        depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
            Depth map
        """
        if is_seq(inv_depth):
            return [inv2depth(item) for item in inv_depth]
        else:
            return 1. * STEREO_SCALE_FACTOR / inv_depth



    def write_depth(self, depth):
        """
        Write a depth map to file, and optionally its corresponding intrinsics.

        This code is modified to export compatible-format depth image to openVSLAM

        Parameters
        ----------
        depth : np.array [H,W]
            Depth map
        """
        # If depth is a tensor
        if is_tensor(depth):
            depth = depth.detach().squeeze().cpu()
            depth = np.clip(depth, 0, 100)

            # make depth image to 16 bit format following TUM RGBD dataset format
            # it is also ROS standard(?)
            depth = np.uint16(depth * 256)  

        return depth

    def cb_image(self, data):
        self.get_logger().info('Received image')
        self.original_input_shape = (data.height, data.width)
        self.process(data)


if __name__ == '__main__':
    hvd_init()
    rclpy.init()
    depth_inference_node = DepthInference()
    rclpy.spin(depth_inference_node)
    depth_inference_node.destroy_node()
    rclpy.shutdown()
