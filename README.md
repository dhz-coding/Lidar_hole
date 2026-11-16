# 隔板空洞体积计算

这个项目按照“目标法向平面分组 → 去掉船舱端部平面组 → 保留内部隔板面 → 投影占据栅格 → 检测封闭空白区域”的流程，估计隔板被挖掉的孔洞体积。

项目默认读取：

```text
E:/work-lidar/base/data/boat2/整体/before/boat2_1_colorized.pcd
```

所有输入、输出和算法参数在 `config.yaml` 中配置。

## 运行

建议使用现有的 `Lidar` Conda 环境：

```powershell
conda activate Lidar
python main.py --config config.yaml
```

当前默认会评估若干候选的孔洞证据，并选择有成对面和封闭空白区域的候选。处理所有候选：

```powershell
python main.py --config config.yaml --candidate-index -1
```

## 算法步骤

1. 从二进制 PCD 头读取字段，使用内存映射和分块读取 XYZ。源文件中的 `rgb`、法向和其他字段不会被改写。
2. 用配置的检测样本和原始法向字段，分别搜索法向接近 x、y 的多组 RANSAC 平面。
3. 按法向、位置、投影重叠把碎片面合并为物理平面组。
4. 自动去掉沿目标法向位置最外侧的端部平面组。若有多个内部组，每个成对且投影重叠的近距离平面被视为一块隔板的前后面。
5. 将隔板面投影到“隔板横向 × 竖直”二维坐标，使用 alpha shape 建立外轮廓。
6. 扫描原始点云在隔板面带内的占据栅格。外轮廓内有实体边界支持的空白连通域作为孔洞候选，是否允许与外轮廓相接由 YAML 控制。
7. 孔洞面积由栅格面积累加，厚度由隔板前后平面间距给出，体积为面积乘厚度。输出同时包含体素化孔洞点云，方便在 CloudCompare 中检查。

## 输出

默认输出到 `output/boat2_1`：

- `plane_detection_report.json`：所有目标轴、RANSAC 平面、平面组和候选排序。
- `hole_volume_report.json`：孔洞面积、隔板厚度、体积和输出文件。
- `candidate_XX_x/partition_face_points.pcd`：候选隔板面点。
- `candidate_XX_x/partition_crop_raw.pcd`：从原始点云截取的隔板附近完整点云，用于人工检查选面。
- `candidate_XX_x/partition_ideal_faces.pcd`：外轮廓扣除孔洞后的理想隔板面。
- `candidate_XX_x/hole_XX_volume_voxels.pcd`：孔洞体积的体素化点云。

## 参数调节顺序

先看 `plane_detection_report.json` 中的平面组位置和 `member_plane_ids`。

如果目标隔板没有被提取，依次检查：

- `normal_angle_deg`
- `ransac_distance_threshold`
- `min_plane_points`
- `max_planes`
- `coordinate.target_normal_axis`

如果前后面被错误合成一个平面组，减小 `grouping.face_merge_distance`。如果前后面没有形成候选，增大 `pair_max_separation` 或减小 `pair_min_overlap`。

如果孔洞被填掉，减小 `hole.closing_radius_cells`，并检查 `hole.grid_size` 是否小于孔洞的最小宽度。如果把外部缺失区域识别成孔洞，优先检查 alpha shape 参数和端部平面组是否被正确去掉。

这份示例点云的 YAML 预设了 `selection.preferred_axis: y` 和 `selection.position_hint: -31.65`，用于把船长方向上的目标隔板与其他横向结构区分开。这个位置提示只对当前坐标系和当前船有效，换数据时应清空或重新设置。

## 结果边界

两张隔板面都能拟合时，厚度是点云测得的。只有一张面时，体积会使用 `single_face_default_thickness`，报告中的 `thickness_source` 会标记为配置值。程序不会把单面厚度伪装成点云实测值。

本次示例运行识别到 y 方向约 `-31.65 m` 的内部平面，检测到 3 个边界相接的空白连通域，按 YAML 中的 `0.08 m` 单面厚度估计总量约 `1.220736 m³`。请先查看 `candidate_06_y/grid_diagnostic.png` 和对应的体素化 PCD，再决定是否接受边界相接区域作为孔洞。报告中的 `touches_outer_boundary` 会保留这个判断依据。

当前默认只保留闭合洞口。空白区域如果接触隔板外轮廓，会被排除，避免把隔板外部空间误生成伪点。

