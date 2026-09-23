import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import trimesh
import trimesh.transformations as tf

# Hardcoded ignores (manually curated)
HARDCODED_IGNORED_IDS = {
    0,
    6,
    7,
    10,
    11,
    33,
    37,
    48,
    55,
    58,
    59,
    60,
    68,
    76,
    84,
    85,
    87,
    102,
    117,
    118,
    123,
    125,
    126,
    132,
    133,
    138,
    143,
    144,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Geometric Graph with OBB Proximity (SAT) - Auto Ignore Largest Instances"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root dir.")
    parser.add_argument("--cluster_dir", type=str, default="cluster_result")
    parser.add_argument("--input_name", type=str, default="atomic_geometry.pt")
    parser.add_argument("--output_name", type=str, default="candidate_graph.pt")

    # === 筛选参数 ===
    parser.add_argument(
        "--ignore_largest_n",
        type=int,
        default=1,
        help="自动忽略体积最大的 N 个物体（通常是墙壁、地板或房间外壳）。设为 0 则不忽略。",
    )

    # === 自适应参数 (相对比例) ===
    parser.add_argument(
        "--proximity_ratio",
        type=float,
        default=0.06,
        help="[Broad Phase] OBB膨胀系数。设置为物体尺寸的 6%。"
             "会给OBB加上这一层的'安全气囊'再做碰撞检测。"
    )
    # 可视化参数
    parser.add_argument(
        "--viz_downsample_num",
        type=int,
        default=1000,
        help="导出 GLB 可视化时每个实例最多采样点数；0 表示不降采样。",
    )

    return parser.parse_args()

def get_characteristic_size(geo_data: dict) -> float:
    """使用 OBB 的最大边长作为特征尺寸，比 AABB 对角线更稳定"""
    # extents 是全长，Open3D/Trimesh 标准
    return float(np.max(geo_data["obb_extent"]))

def check_obb_intersection_sat(center_a, extent_a, rot_a,
                               center_b, extent_b, rot_b,
                               margin=0.0) -> bool:
    """
    使用分离轴定理 (SAT) 检测两个 OBB 是否接近。
    我们在检测前给 Extent 加上 margin，相当于检测 '膨胀后的 OBB' 是否碰撞。
    """
    # 1. 准备数据 (SAT 使用半长 half-extents)
    # 加上 margin 作为安全气囊
    ea = (extent_a / 2.0) + margin
    eb = (extent_b / 2.0) + margin

    # 平移向量 (在世界坐标系)
    t = center_b - center_a

    # 旋转矩阵 (列向量是局部轴)
    # Ra[:, 0] 是 A 的 x 轴, Ra[:, 1] 是 y 轴...
    Ra = rot_a
    Rb = rot_b

    # 计算旋转矩阵的点积矩阵 (用于投影)
    # R[i, j] = dot(Ra_i, Rb_j)
    R = np.dot(Ra.T, Rb)

    # 为了数值稳定性，加上绝对值
    AbsR = np.abs(R) + 1e-6

    # === SAT 检测 ===
    # 我们需要检测 15 个轴。只要有一个轴上投影不重叠，就返回 False (不相交)。

    # Case 1: 3个轴是 A 的基向量 (A's basis vectors)
    # Project onto Ra_x, Ra_y, Ra_z
    # t 在 Ra 坐标系下的投影
    t_in_a = np.dot(t, Ra)

    for i in range(3):
        # 投影半径 A = ea[i]
        # 投影半径 B = dot(eb, AbsR[i, :])
        ra = ea[i]
        rb = np.dot(eb, AbsR[i, :])
        if np.abs(t_in_a[i]) > (ra + rb):
            return False # 分离

    # Case 2: 3个轴是 B 的基向量 (B's basis vectors)
    # Project onto Rb_x, Rb_y, Rb_z
    t_in_b = np.dot(t, Rb)

    for i in range(3):
        # 投影半径 A = dot(ea, AbsR[:, i])
        # 投影半径 B = eb[i]
        ra = np.dot(ea, AbsR[:, i])
        rb = eb[i]
        if np.abs(t_in_b[i]) > (ra + rb):
            return False # 分离

    # Case 3: 9个轴是 A 和 B 基向量的叉积 (Cross products)
    # 这是检测 Edge-Edge 碰撞的关键，也是 OBB 比 AABB 准的原因

    # Axis: Ra_x X Rb_x
    ra = ea[1] * AbsR[2, 0] + ea[2] * AbsR[1, 0]
    rb = eb[1] * AbsR[0, 2] + eb[2] * AbsR[0, 1]
    if np.abs(t_in_a[2] * R[1, 0] - t_in_a[1] * R[2, 0]) > (ra + rb): return False

    # Axis: Ra_x X Rb_y
    ra = ea[1] * AbsR[2, 1] + ea[2] * AbsR[1, 1]
    rb = eb[0] * AbsR[0, 2] + eb[2] * AbsR[0, 0]
    if np.abs(t_in_a[2] * R[1, 1] - t_in_a[1] * R[2, 1]) > (ra + rb): return False

    # Axis: Ra_x X Rb_z
    ra = ea[1] * AbsR[2, 2] + ea[2] * AbsR[1, 2]
    rb = eb[0] * AbsR[0, 1] + eb[1] * AbsR[0, 0]
    if np.abs(t_in_a[2] * R[1, 2] - t_in_a[1] * R[2, 2]) > (ra + rb): return False

    # Axis: Ra_y X Rb_x
    ra = ea[0] * AbsR[2, 0] + ea[2] * AbsR[0, 0]
    rb = eb[1] * AbsR[1, 2] + eb[2] * AbsR[1, 1]
    if np.abs(t_in_a[0] * R[2, 0] - t_in_a[2] * R[0, 0]) > (ra + rb): return False

    # Axis: Ra_y X Rb_y
    ra = ea[0] * AbsR[2, 1] + ea[2] * AbsR[0, 1]
    rb = eb[0] * AbsR[1, 2] + eb[2] * AbsR[1, 0]
    if np.abs(t_in_a[0] * R[2, 1] - t_in_a[2] * R[0, 1]) > (ra + rb): return False

    # Axis: Ra_y X Rb_z
    ra = ea[0] * AbsR[2, 2] + ea[2] * AbsR[0, 2]
    rb = eb[0] * AbsR[1, 1] + eb[1] * AbsR[1, 0]
    if np.abs(t_in_a[0] * R[2, 2] - t_in_a[2] * R[0, 2]) > (ra + rb): return False

    # Axis: Ra_z X Rb_x
    ra = ea[0] * AbsR[1, 0] + ea[1] * AbsR[0, 0]
    rb = eb[1] * AbsR[2, 2] + eb[2] * AbsR[2, 1]
    if np.abs(t_in_a[1] * R[0, 0] - t_in_a[0] * R[1, 0]) > (ra + rb): return False

    # Axis: Ra_z X Rb_y
    ra = ea[0] * AbsR[1, 1] + ea[1] * AbsR[0, 1]
    rb = eb[0] * AbsR[2, 2] + eb[2] * AbsR[2, 0]
    if np.abs(t_in_a[1] * R[0, 1] - t_in_a[0] * R[1, 1]) > (ra + rb): return False

    # Axis: Ra_z X Rb_z
    ra = ea[0] * AbsR[1, 2] + ea[1] * AbsR[0, 2]
    rb = eb[0] * AbsR[2, 1] + eb[1] * AbsR[2, 0]
    if np.abs(t_in_a[1] * R[0, 2] - t_in_a[0] * R[1, 2]) > (ra + rb): return False

    # 如果所有 15 个轴都有重叠，则判定 OBB 相交（在 margin 范围内）
    return True

def export_graph_viz(
    geometries,
    edges,
    output_path: Path,
    edge_color_rgba=(255, 0, 0, 255),
    viz_downsample=0,
    exclude_nodes=None,
):
    """
    生成可视化 GLB：
    1. 渲染所有物体的点云（灰色作为背景，活跃节点彩色）。
    2. 在相邻物体之间画红色的连接线（Cylinders）。
    """
    print(f"[Viz] Generating graph visualization: {output_path}")
    scene = trimesh.Scene()
    exclude_nodes = set() if exclude_nodes is None else set(exclude_nodes)

    # 1. 找出所有“活跃”的节点（即有边相连的节点）
    active_nodes = set()
    for e in edges:
        active_nodes.add(e["u"])
        active_nodes.add(e["v"])

    # 2. 绘制所有物体的点云（可选降采样以减小文件体积）
    for uid, geo in geometries.items():
        if uid in exclude_nodes:
            continue
        pts = geo["geometry_points"]
        if viz_downsample and len(pts) > viz_downsample:
            rng = np.random.default_rng(int(uid))
            pts = pts[rng.choice(len(pts), viz_downsample, replace=False)]

        # 颜色策略：有关系的物体给彩色，孤立物体给淡灰色
        if uid in active_nodes:
            rng = np.random.default_rng(int(uid))
            color = np.append(rng.integers(50, 255, 3), 255).astype(np.uint8)
        else:
            color = np.array([200, 200, 200, 50], dtype=np.uint8)

        colors = np.tile(color, (len(pts), 1))
        pcd = trimesh.points.PointCloud(pts, colors=colors)
        scene.add_geometry(pcd)

    # 3. 绘制连接线 (Edges)
    edge_color = np.array(edge_color_rgba, dtype=np.uint8)

    # 自适应线粗细：按场景中位物体尺寸的 1% 作为半径（无尺寸时回退到 0.005）
    sizes = []
    for geo in geometries.values():
        try:
            sizes.append(float(np.max(geo["obb_extent"])))
        except Exception:
            continue
    median_size = float(np.median(sizes)) if len(sizes) else 0.0
    radius = (median_size * 0.01) if np.isfinite(median_size) and median_size > 0 else 0.005
    radius = max(float(radius), 1e-6)

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for e in edges:
        u, v = e["u"], e["v"]
        if u in exclude_nodes or v in exclude_nodes:
            continue
        p1 = np.asarray(geometries[u]["obb_center"], dtype=np.float64)
        p2 = np.asarray(geometries[v]["obb_center"], dtype=np.float64)

        # 创建一个细圆柱体连接 p1 和 p2
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue

        v_dir = vec / length

        # Trimesh cylinder 默认沿 Z 轴创建、中心在原点：旋转到 vec 方向，再平移到中点
        if np.allclose(v_dir, z_axis, atol=1e-6):
            R = np.eye(4)
        elif np.allclose(v_dir, -z_axis, atol=1e-6):
            R = tf.rotation_matrix(np.pi, [1, 0, 0])
        else:
            axis = np.cross(z_axis, v_dir)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-12:
                R = np.eye(4)
            else:
                axis = axis / axis_norm
                angle = float(np.arccos(np.clip(np.dot(z_axis, v_dir), -1.0, 1.0)))
                R = tf.rotation_matrix(angle, axis)

        midpoint = (p1 + p2) / 2.0
        T = tf.translation_matrix(midpoint)

        cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
        cylinder.visual.face_colors = edge_color
        cylinder.apply_transform(T @ R)
        scene.add_geometry(cylinder)

    # 4. 导出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    print(f"[Viz] Saved to {output_path}")

def _seven_seg_digit_mesh(digit: str, seg_len: float, seg_thick: float, depth: float) -> trimesh.Trimesh:
    """
    Build a single 7-segment digit mesh centered roughly around origin on XY plane.
    Segment layout (A-G):
      A
    F   B
      G
    E   C
      D
    """
    # Segment centers in local digit frame
    half = seg_len * 0.5
    off = half + seg_thick * 0.5

    # extents for horizontal/vertical segments
    h_dim = np.array([seg_len, seg_thick, depth], dtype=np.float64)
    v_dim = np.array([seg_thick, seg_len, depth], dtype=np.float64)

    segs = {}
    segs["A"] = (h_dim, np.array([0.0, +off, 0.0]))
    segs["D"] = (h_dim, np.array([0.0, -off, 0.0]))
    segs["G"] = (h_dim, np.array([0.0, 0.0, 0.0]))
    segs["F"] = (v_dim, np.array([-off, +half * 0.5, 0.0]))
    segs["B"] = (v_dim, np.array([+off, +half * 0.5, 0.0]))
    segs["E"] = (v_dim, np.array([-off, -half * 0.5, 0.0]))
    segs["C"] = (v_dim, np.array([+off, -half * 0.5, 0.0]))

    on = set()
    if digit == "0":
        on = {"A", "B", "C", "D", "E", "F"}
    elif digit == "1":
        on = {"B", "C"}
    elif digit == "2":
        on = {"A", "B", "G", "E", "D"}
    elif digit == "3":
        on = {"A", "B", "C", "D", "G"}
    elif digit == "4":
        on = {"F", "G", "B", "C"}
    elif digit == "5":
        on = {"A", "F", "G", "C", "D"}
    elif digit == "6":
        on = {"A", "F", "E", "D", "C", "G"}
    elif digit == "7":
        on = {"A", "B", "C"}
    elif digit == "8":
        on = {"A", "B", "C", "D", "E", "F", "G"}
    elif digit == "9":
        on = {"A", "B", "C", "D", "F", "G"}
    elif digit == "-":
        on = {"G"}

    meshes = []
    for name in sorted(on):
        dim, trans = segs[name]
        mat = tf.translation_matrix(trans)
        meshes.append(trimesh.creation.box(extents=dim, transform=mat))
    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes)

def _seven_seg_text_mesh(text: str, seg_len: float, seg_thick: float, depth: float, spacing: float) -> trimesh.Trimesh:
    meshes = []
    cursor = 0.0
    for ch in str(text):
        digit_mesh = _seven_seg_digit_mesh(ch, seg_len=seg_len, seg_thick=seg_thick, depth=depth)
        mat = tf.translation_matrix([cursor, 0.0, 0.0])
        digit_mesh.apply_transform(mat)
        meshes.append(digit_mesh)
        cursor += seg_len + spacing
    if not meshes:
        return trimesh.Trimesh()
    combined = trimesh.util.concatenate(meshes)
    # center horizontally
    mat_center = tf.translation_matrix([-cursor * 0.5 + (seg_len + spacing) * 0.5, 0.0, 0.0])
    combined.apply_transform(mat_center)
    return combined

def export_instance_id_viz(
    geometries,
    output_path: Path,
    viz_downsample=0,
    exclude_nodes=None,
):
    """
    生成实例 ID 可视化 GLB：
    - 点云：显示实例形态
    - ID：在每个实例 OBB 顶部上方放置 7 段数码管风格的 3D 标签（无需字体依赖）
    """
    print(f"[Viz] Generating instance-id visualization: {output_path}")
    scene = trimesh.Scene()
    exclude_nodes = set() if exclude_nodes is None else set(exclude_nodes)

    sizes = []
    for uid, geo in geometries.items():
        if uid in exclude_nodes:
            continue
        try:
            sizes.append(float(np.max(geo["obb_extent"])))
        except Exception:
            continue
    median_size = float(np.median(sizes)) if len(sizes) else 0.0
    seg_len = (median_size * 0.12) if np.isfinite(median_size) and median_size > 0 else 0.05
    seg_len = max(float(seg_len), 1e-6)
    seg_thick = max(seg_len * 0.22, 1e-6)
    depth = seg_thick
    spacing = seg_len * 0.55

    label_color = np.array([255, 255, 0, 255], dtype=np.uint8)

    for uid, geo in geometries.items():
        if uid in exclude_nodes:
            continue

        pts = geo["geometry_points"]
        if viz_downsample and len(pts) > viz_downsample:
            rng = np.random.default_rng(int(uid))
            pts = pts[rng.choice(len(pts), viz_downsample, replace=False)]

        colors = np.tile(np.array([180, 180, 180, 120], dtype=np.uint8), (len(pts), 1))
        pcd = trimesh.points.PointCloud(pts, colors=colors)
        scene.add_geometry(pcd, geom_name=f"pcd_{uid}")

        center = np.asarray(geo["obb_center"], dtype=np.float64)
        label_pos = center.copy()

        text_mesh = _seven_seg_text_mesh(str(uid), seg_len=seg_len, seg_thick=seg_thick, depth=depth, spacing=spacing)
        if len(text_mesh.vertices) == 0:
            continue
        text_mesh.visual.face_colors = label_color
        T = tf.translation_matrix(label_pos)
        text_mesh.apply_transform(T)
        scene.add_geometry(text_mesh, geom_name=f"id_{uid}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    print(f"[Viz] Saved to {output_path}")

def main():
    args = parse_args()

    # 1. Load Data
    data_path = Path(args.data_dir) / args.cluster_dir / args.input_name
    print(f"[Graph] Loading atomic geometry: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    geometries = torch.load(data_path)
    instance_ids = sorted(list(geometries.keys()))

    # === [新增] 自动识别并忽略体积最大的物体（通常是墙/地板/房间外壳） ===
    ignored_ids = set()
    hard_ignored_ids = sorted(list(set(instance_ids).intersection(HARDCODED_IGNORED_IDS)))
    if len(hard_ignored_ids) > 0:
        ignored_ids.update(hard_ignored_ids)
        print(f"\n[Filter] Hardcoded ignored instances: {hard_ignored_ids}")

    if args.ignore_largest_n > 0 and len(instance_ids) > 0:
        volumes = []
        for uid in instance_ids:
            ext = np.asarray(geometries[uid].get("obb_extent", [0.0, 0.0, 0.0]), dtype=np.float64)
            vol = float(np.prod(np.maximum(ext, 0.0)))
            if np.isfinite(vol):
                volumes.append((uid, vol))

        volumes.sort(key=lambda x: x[1], reverse=True)
        top_n = min(len(volumes), int(args.ignore_largest_n))
        if top_n > 0:
            print(f"\n[Filter] Identifying top {top_n} largest instances (Walls/Floor):")
            for i in range(top_n):
                uid, vol = volumes[i]
                ignored_ids.add(uid)
                print(f"  - IGNORE Instance {uid}: Volume = {vol:.6f}")
    # ======================================================================

    valid_ids = [uid for uid in instance_ids if uid not in ignored_ids]
    if len(valid_ids) == 0:
        raise ValueError(
            f"No instances left after filtering (ignore_largest_n={args.ignore_largest_n})."
        )

    # 2. 预计算尺寸
    sizes = {}
    print("[Graph] Pre-calculating object characteristic sizes...")
    for uid in valid_ids:
        sizes[uid] = get_characteristic_size(geometries[uid])

    print(f"  Avg Size (valid): {np.mean(list(sizes.values())):.4f}")

    edges = []
    broad_phase_pass = 0

    # 3. 遍历检测
    total_pairs = len(valid_ids) * (len(valid_ids) - 1) // 2
    pbar = tqdm(total=total_pairs, desc="OBB Proximity Check")

    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            uid_a = valid_ids[i]
            uid_b = valid_ids[j]
            geo_a = geometries[uid_a]
            geo_b = geometries[uid_b]

            pbar.update(1)

            # 动态阈值
            size_a = sizes[uid_a]
            size_b = sizes[uid_b]
            min_size = min(size_a, size_b)

            dynamic_margin = min_size * args.proximity_ratio

            # --- Broad Phase: OBB SAT ---
            # 这里的 margin 实际上是把 OBB 变大了一圈
            if not check_obb_intersection_sat(
                geo_a["obb_center"], geo_a["obb_extent"], geo_a["obb_rotation"],
                geo_b["obb_center"], geo_b["obb_extent"], geo_b["obb_rotation"],
                margin=dynamic_margin
            ):
                continue

            broad_phase_pass += 1
            edges.append(
                {
                    "u": uid_a,
                    "v": uid_b,
                    "margin_used": dynamic_margin,
                    "info": f"OBB SAT pass (margin={dynamic_margin:.4f})",
                }
            )

    pbar.close()

    # 4. Save
    graph_data = {
        "nodes": instance_ids,
        "valid_nodes": valid_ids,
        "ignored_nodes": sorted(list(ignored_ids)),
        "hard_ignored_nodes": hard_ignored_ids,
        "edges": edges,
        "metadata": {
            "method": "obb_sat_ignore_largest",
            "proximity_ratio": args.proximity_ratio,
            "ignore_largest_n": int(args.ignore_largest_n),
            "hard_ignored_count": int(len(hard_ignored_ids)),
            "stats": {
                "total_pairs": total_pairs,
                "passed_broad_phase": broad_phase_pass,
                "final_edges": len(edges)
            }
        }
    }

    out_path = Path(args.data_dir) / args.cluster_dir / args.output_name
    torch.save(graph_data, out_path)

    # === [新增] 图可视化：点云 + 邻接红线 ===
    viz_path = Path(args.data_dir) / args.cluster_dir / "graph_viz.glb"
    export_graph_viz(
        geometries,
        edges,
        viz_path,
        edge_color_rgba=(255, 0, 0, 255),
        viz_downsample=args.viz_downsample_num,
        exclude_nodes=ignored_ids,
    )

    # === [新增] 实例 ID 可视化：点云 + 7 段数字标签 ===
    id_viz_path = Path(args.data_dir) / args.cluster_dir / "instance_id_viz.glb"
    export_instance_id_viz(
        geometries,
        id_viz_path,
        viz_downsample=args.viz_downsample_num,
        exclude_nodes=None,
    )

    print(f"\n[Done] OBB Graph construction finished.")
    print(f"  Ignored Instances: {sorted(list(ignored_ids))}")
    print(f"  Total Pairs: {total_pairs}")
    print(f"  Passed OBB Check: {broad_phase_pass} (Recall Candidates)")
    print(f"  Final Edges: {len(edges)} (Precision Filtering)")
    if len(edges) > 0:
        print("[Preview]")
        print(f"  {edges[0]['u']} <-> {edges[0]['v']}: {edges[0]['info']}")

    print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()
