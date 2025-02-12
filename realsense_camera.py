import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
    def start(self):
        """启动相机"""
        self.pipeline.start(self.config)
        
    def get_frames(self):
        """获取帧"""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET)
        
        return {
            'depth_frame': depth_frame,
            'color_frame': color_frame,
            'depth_image': depth_image,
            'color_image': color_image,
            'depth_colormap': depth_colormap,
            'intrinsics': depth_frame.profile.as_video_stream_profile().intrinsics
        }
    
    def save_point_cloud(self, depth_frame, color_frame, filename="object.ply"):
        """保存点云数据"""
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        points.export_to_ply(filename, color_frame)
        
    def stop(self):
        """停止相机"""
        self.pipeline.stop()
