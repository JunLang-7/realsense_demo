import open3d as o3d

def visualize_ply(ply_path):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(ply_path)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # 修改为你的PLY文件路径
    ply_file = "object.ply"
    visualize_ply(ply_file)
