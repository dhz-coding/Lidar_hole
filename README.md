# 采砂船横梁洞口体积计算

这个项目使用装沙前配准点云计算横梁或隔板洞口的近似体积。当前流程不预设隔板法向与全局 x、y、z 轴的关系，而是从点云中提取全部主要平面，再根据平面法向、面间距和投影重叠配对横梁的两张表面。

## 运行

建议使用现有的 `Lidar` Conda 环境：

```powershell
conda activate Lidar
python main.py --config config.yaml
```

配置文件负责输入点云、输出目录和全部算法参数。

## 算法步骤

1. 从二进制 PCD 读取 XYZ，并通过内存映射、抽样和分块扫描控制内存占用。
2. 使用 RANSAC 在全部方向搜索主要平面，不按全局 x、y 轴预筛选。
3. 将法向相似、位置接近且投影范围重叠的平面碎片合并为物理平面组。
4. 对平面组两两配对。配对条件包括法向夹角、平面间距 `Wmin～Wmax` 和二维投影重叠率。
5. 对每个配对平面使用原始点云做 TLS/SVD 精拟合，得到平面间距 `W`。
6. 根据配对平面法向和两面点云的主方向建立局部坐标系，沿法向投影到二维截面。
7. 将二维截面切成栅格，使用有点或无点判断实体区域和空白区域，再用连通域分析识别闭合洞口。
8. 通过面积、边界支持、矩形填充率、凸包实体度和长宽比排除小缺口、裂缝和严重不规则区域。
9. 对保留洞口使用 `V = L × H × W` 计算近似长方体体积，同时输出伪点云、长方体顶点和日志。

## 洞口筛选

当前默认参数如下：

- `exclude_area_m2: 1.0`，面积小于等于 1 m² 的洞不参与最终体积。
- `min_rectangularity: 0.60`，实际栅格面积与外接矩形面积的比例过低时排除。
- `min_solidity: 0.60`，实际面积与凸包面积的比例过低时排除。
- `max_aspect_ratio: 8.0`，过于细长的空白区域排除。
- `reject_touching_outer_boundary: true`，连接外轮廓的开口排除。

日志中的 `hole_count` 是基础连通域候选数量，`kept_hole_count` 是通过面积和形状筛选后参与体积计算的数量。每个保留洞口还会记录 `rectangularity`、`solidity`、`aspect_ratio` 和 `boundary_support_ratio`。

## 输出

每次运行会在 `paths.output_dir` 下创建时间戳和代码版本目录，默认保留：

- `partition_face_01.pcd` 和 `partition_face_02.pcd`，配对平面点云。
- `partition_plane_group.pcd`，候选横梁平面组点云。
- `partition_slab_points.pcd`，横梁附近原始裁剪点云。
- `hole1_pseudo_points.pcd`、`hole2_pseudo_points.pcd`，洞口伪体积点云。
- `hole1_cuboid_vertices.pcd`、`hole2_cuboid_vertices.pcd`，洞口近似长方体的 8 个顶点。
- `run.log`，参数、配对法向、厚度、洞口尺寸、形状指标、体积和耗时。

## 结果边界

两张平面都成功拟合时，宽度 `W` 来自原始点云精拟合后的面间距。体积方法 `cuboid_bbox` 使用洞口二维外接矩形的长度和高度，因此不规则洞口会有近似误差。形状筛选用于减少误检，不能替代真实洞口轮廓积分。
