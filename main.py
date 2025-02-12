import cv2
from realsense_camera import RealSenseCamera
from detector import ObjectDetector
from cloud_processor import CloudProcessor

class GraspSystem:
    """
    使用方法：
    直接运行main.py启动程序
    按'v'键切换点云可视化
    按'ESC'键退出程序
    """
    def __init__(self):
        self.camera = RealSenseCamera()
        self.detector = ObjectDetector()
        self.cloud_processor = CloudProcessor()
        self.visualization_started = False
        
    def run(self):
        """运行主循环"""
        self.camera.start()
        
        try:
            while True:
                # 获取相机数据
                frames = self.camera.get_frames()
                
                # 目标检测
                detections = self.detector.detect(frames['color_image'])
                color_image = self.detector.draw_detections(frames['color_image'], detections)
                
                # 点云处理
                pcd = self.cloud_processor.create_point_cloud(
                    frames['depth_image'], 
                    frames['intrinsics'])
                
                # 平面分割
                plane_cloud, objects_cloud, _ = self.cloud_processor.segment_plane(pcd)
                clusters = self.cloud_processor.cluster_objects(objects_cloud)
                
                # 点云可视化
                if len(clusters) > 0:
                    visualize_clouds = [plane_cloud] + clusters
                    if not self.visualization_started:
                        self.cloud_processor.start_visualization_thread(visualize_clouds)
                        self.visualization_started = True
                    else:
                        self.cloud_processor.visualize_point_clouds(visualize_clouds)
                
                # 显示物体信息
                for _, det in detections.iterrows():
                    obj_info = self.detector.get_object_3d_position(
                        det, frames['depth_frame'], frames['intrinsics'])
                    pos = obj_info['position']
                    print(f"物体: {obj_info['object']}, "
                          f"置信度: {obj_info['confidence']:.2f}, "
                          f"3D位置 (x,y,z): ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) 米")
                
                # 显示图像
                cv2.imshow('RGB', color_image)
                cv2.imshow('Depth', frames['depth_colormap'])
                cv2.moveWindow('RGB', 0, 0)
                cv2.moveWindow('Depth', 640, 0)
                
                # 保存点云
                self.camera.save_point_cloud(
                    frames['depth_frame'], 
                    frames['color_frame'])
                
                # 键盘控制
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord('v'):  # 切换点云可视化
                    if self.visualization_started:
                        self.cloud_processor.stop_visualization()
                        self.visualization_started = False
                    else:
                        self.cloud_processor.start_visualization_thread([pcd])
                        self.visualization_started = True
                        
        finally:
            self.cleanup()
            
    def cleanup(self):
        """清理资源"""
        if self.visualization_started:
            self.cloud_processor.stop_visualization()
        self.camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = GraspSystem()
    system.run()
