import torch
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name='yolo11n'):
        # 加载YOLO模型
        self.model = YOLO(model_name)
        
    def detect(self, color_image):
        """执行目标检测"""
        results = self.model(color_image)
        
        # 检查 results 是否为列表
        if isinstance(results, list):
            # 假设 results 列表中的第一个元素包含检测结果
            if len(results) > 0:
                # 获取 Boxes 对象
                boxes = results[0].boxes
                # 将 Boxes 对象转换为 pandas DataFrame
                detections = boxes.data.cpu().numpy()
                # 创建 DataFrame
                df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
                return df
            else:
                return pd.DataFrame()  # 返回一个空的 DataFrame
        else:
            # 如果 results 不是列表，则尝试直接访问 boxes 属性
            try:
                # 获取 Boxes 对象
                boxes = results.boxes
                # 将 Boxes 对象转换为 pandas DataFrame
                detections = boxes.data.cpu().numpy()
                # 创建 DataFrame
                df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
                return df
            except AttributeError as e:
                print(f"Error: 'Results' object does not have 'boxes' attribute. {e}")
                return pd.DataFrame()  # 返回一个空的 DataFrame
    
    def draw_detections(self, image, detections):
        """在图像上绘制检测结果"""
        if detections is not None and not detections.empty:
            for _, det in detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf, cls_name = det['confidence'], det['class']
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{cls_name}: {conf:.2f}'
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image

    def get_object_3d_position(self, detection, depth_frame, depth_intrin):
        """计算检测到的物体的3D位置"""
        x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], 
                                  detection['xmax'], detection['ymax']])
        
        # 使用边界框中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        depth = depth_frame.get_distance(center_x, center_y)
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], depth)
        
        return {
            'object': detection['class'],
            'confidence': detection['confidence'],
            'position': point_3d,
            'pixel_position': (center_x, center_y)
        }
