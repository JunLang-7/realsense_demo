import open3d as o3d
import numpy as np
import threading
import time

class CloudProcessor:
    def __init__(self):
        self.vis = None
        self.is_visualizing = False
        self.visualization_lock = threading.Lock()
    
    def _initialize_visualizer(self):
        """初始化可视化器"""
        try:
            if self.vis is None:
                self.vis = o3d.visualization.Visualizer()
                # 创建窗口时设置位置和大小 (x, y, width, height)
                success = self.vis.create_window(width=640, height=480, left=1280, top=100)
                if not success:
                    print("创建可视化窗口失败")
                    return False
                    
                # 设置渲染选项
                render_option = self.vis.get_render_option()
                render_option.point_size = 2.0
                render_option.background_color = np.array([0.1, 0.1, 0.1])
                # 设置默认视角
                view_control = self.vis.get_view_control()
                view_control.set_zoom(0.8)
                view_control.set_front([0, 0, -1])
                view_control.set_lookat([0, 0, 0])
                view_control.set_up([0, -1, 0])
                return True
        except Exception as e:
            print(f"初始化可视化器失败: {e}")
            return False

    def create_point_cloud(self, depth_image, depth_intrin):
        """从深度图像创建点云"""
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_image),
            o3d.camera.PinholeCameraIntrinsic(
                width=depth_intrin.width,
                height=depth_intrin.height,
                fx=depth_intrin.fx,
                fy=depth_intrin.fy,
                cx=depth_intrin.ppx,
                cy=depth_intrin.ppy
            )
        )
        # 移除无效点
        pcd = self.preprocess_point_cloud(pcd)
        return pcd

    def preprocess_point_cloud(self, pcd, voxel_size=0.02):
        """点云预处理：降采样和移除异常点"""
        # 移除无效点
        pcd = pcd.remove_non_finite_points()
        # 体素下采样
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        # 移除统计异常点
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd.select_by_index(ind)

    def segment_plane(self, pcd, distance_threshold=0.02, ransac_n=3, num_iterations=100):
        """分割平面（例如桌面）"""
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        plane_cloud = pcd.select_by_index(inliers)
        other_cloud = pcd.select_by_index(inliers, invert=True)
        return plane_cloud, other_cloud, plane_model

    def cluster_objects(self, pcd, eps=0.02, min_points=100):
        """对点云进行聚类分割"""
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        max_label = labels.max()
        clusters = []
        
        # 为每个聚类创建点云对象
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) >= min_points:
                cluster = pcd.select_by_index(cluster_indices)
                clusters.append(cluster)
        
        return clusters

    def visualize_point_clouds(self, clouds):
        """可视化一个或多个点云"""
        with self.visualization_lock:
            if not isinstance(clouds, list):
                clouds = [clouds]
            
            if self.vis is None:
                return

            try:
                # 清空可视化器
                self.vis.clear_geometries()
                
                # 为不同的点云设置不同的颜色
                colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], 
                         [1, 1, 0], [1, 0, 1], [0, 1, 1]]
                for i, cloud in enumerate(clouds):
                    if not cloud.has_colors():
                        cloud.paint_uniform_color(colors[i % len(colors)])
                    self.vis.add_geometry(cloud)

                # 更新渲染
                self.vis.poll_events()
                self.vis.update_renderer()
            except Exception as e:
                print(f"可视化更新失败: {e}")
    
    def start_visualization_thread(self, clouds):
        """在单独的线程中启动可视化"""
        if not self.is_visualizing:
            self.is_visualizing = True
            try:
                self._initialize_visualizer()
                vis_thread = threading.Thread(target=self._run_visualization, 
                                           args=(clouds,))
                vis_thread.daemon = True
                vis_thread.start()
            except Exception as e:
                print(f"启动可视化线程失败: {e}")
                self.is_visualizing = False
    
    def _run_visualization(self, clouds):
        """运行可视化循环"""
        try:
            self.visualize_point_clouds(clouds)
            while self.is_visualizing:
                with self.visualization_lock:
                    if self.vis is not None:
                        self.vis.poll_events()
                        self.vis.update_renderer()
                time.sleep(0.1)  # 添加小延时，减少 CPU 占用
        except Exception as e:
            print(f"可视化循环出错: {e}")
        finally:
            self.is_visualizing = False
            
    def stop_visualization(self):
        """停止可视化"""
        self.is_visualizing = False
        with self.visualization_lock:
            if self.vis is not None:
                try:
                    self.vis.destroy_window()
                except Exception as e:
                    print(f"关闭可视化窗口失败: {e}")
                self.vis = None
