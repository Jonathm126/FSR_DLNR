from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import plotly.graph_objects as go

# ---------------------------
# Visualization helpers
# ---------------------------
def visualize_disparity(disp, ref_img, valid_mask=None, show=True):
    """Normalize disparity on valid values and show side-by-side with reference image."""
    if valid_mask is None:
        valid_mask = np.isfinite(disp)

    # Normalize only valid disparities
    disp_valid = np.where(valid_mask, disp, np.nan)
    min_v, max_v = np.nanmin(disp_valid), np.nanmax(disp_valid)
    disp_norm = np.zeros_like(disp_valid, dtype=np.float32) if np.isclose(min_v, max_v) \
                else (disp_valid - min_v) / (max_v - min_v)
    disp_uint8 = (np.nan_to_num(disp_norm) * 255).astype(np.uint8)

    # Apply colormap
    disp_color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_INFERNO)
    disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)

    # Side-by-side composition
    h, w = ref_img.shape[:2]
    ref_rgb = ref_img if ref_img.ndim == 3 else cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
    combined = Image.new('RGB', (2 * w, h))
    combined.paste(Image.fromarray(ref_rgb), (0, 0))
    combined.paste(Image.fromarray(disp_color), (w, 0))

    if show:
        combined.show()

    return combined, disp_uint8


def plot_points_plotly(points_xyz, colors_rgb=None, title="Point Cloud"):
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]

    # Center + equal scale cube
    cx, cy, cz = np.mean(x), np.mean(y), np.mean(z)
    r = max(np.ptp(x), np.ptp(y), np.ptp(z)) / 2.0
    xr, yr, zr = [cx - r, cx + r], [cy - r, cy + r], [cz - r, cz + r]

    marker_kw = dict(size=2, opacity=0.9)
    if colors_rgb is not None:
        marker_kw["color"] = colors_rgb  # Nx3 floats in [0,1]

    fig = go.Figure(
        data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker_kw)]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=xr, title="X", showbackground=True, backgroundcolor="rgb(240,240,240)"),
            yaxis=dict(range=yr, title="Y", showbackground=True, backgroundcolor="rgb(240,240,240)"),
            zaxis=dict(range=zr, title="Z", showbackground=True, backgroundcolor="rgb(240,240,240)"),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=800
    )
    fig.show()

# ---------------------------
# Stereo + 3D reconstruction
# ---------------------------
def preprocess_gray(img, target_short_side=720, clahe=True):
    """Resize (keeping aspect) so min(h,w)=target_short_side; optional CLAHE."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure dtype is uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = img.shape[:2]
    scale = float(target_short_side) / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = c.apply(out)

    return out

def make_sgbm(min_disp, max_disp, block_size, uniqueness, speckle_win, speckle_range, block_factor):
    """Create an OpenCV StereoSGBM matcher with valid parameters."""
    num_disp = max_disp - min_disp
    if num_disp <= 0:
        raise ValueError("max_disp must be greater than min_disp")
    # numDisparities must be divisible by 16
    num_disp = int(np.ceil(num_disp / 16.0)) * 16

    P1 = int(8 * block_factor * block_size ** 2)
    P2 = int(32 * block_factor * block_size ** 2)

    sgbm = cv2.StereoSGBM.create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=0,
        uniquenessRatio=uniqueness,
        speckleWindowSize=speckle_win,
        speckleRange=speckle_range,
        mode=3
    )
    return sgbm

def disparity_to_points(disp, img, Q, valid_mask, z_clip=None, flip_z=False):
    "Projects disparity image to 3d points, returns points and point colors"
    pts3d = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
    if flip_z:
        pts3d[..., 2] *= -1

    if img.ndim == 2:
        colors = np.repeat(img[..., None], 3, axis=2) / 255.0
    else:
        colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    points = pts3d[valid_mask]
    cols = colors[valid_mask]

    if z_clip is not None:
        zmin, zmax = z_clip
        keep = (points[:, 2] > zmin) & (points[:, 2] < zmax)
        points, cols = points[keep], cols[keep]

    finite = np.isfinite(points).all(axis=1)
    return points[finite], cols[finite]

# ---------------------------
# Open3D utilities
# ---------------------------
def to_o3d(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd

def downsample_and_denoise(pcd, voxel, nb_neighbors, std_ratio):
    """voxel in scene units (depends on Q); tune as needed."""
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel)
    pcd_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30)
    )
    pcd_clean, _ = pcd_ds.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_clean

def compute_fpfh(pcd, voxel):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
    )

def register_pair(source, target, voxel=0.01):
    """Coarse RANSAC + fine ICP."""
    s_down = source.voxel_down_sample(voxel)
    t_down = target.voxel_down_sample(voxel)
    s_feat = compute_fpfh(s_down, voxel)
    t_feat = compute_fpfh(t_down, voxel)

    distance_threshold = voxel * 1.5

    # Coarse global alignment (RANSAC)
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s_down, t_down, s_feat, t_feat, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    # Refine with ICP
    icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold,
        ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return icp.transformation

def multiway_register(pcd_list, voxel=0.01):
    """Pose-graph global registration + optimization (Open3D)."""
    if len(pcd_list) == 1:
        return [np.eye(4)]

    # Initialize pose graph
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.eye(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry.copy()))
    edges = []

    # Odometric edges (sequential)
    for i in range(1, len(pcd_list)):
        Ti = register_pair(pcd_list[i], pcd_list[i-1], voxel=voxel)  # i -> i-1
        odometry = Ti @ odometry
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pcd_list[i], pcd_list[i-1], max_correspondence_distance=voxel * 1.5, transformation=Ti
        )
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i-1, i, Ti, info, uncertain=False))
        edges.append(((i-1, i), Ti))

    # Optional loop closures (sparse): here connect every 3rd frame
    for i in range(len(pcd_list)):
        for j in range(i+3, len(pcd_list), 3):
            Tij = register_pair(pcd_list[j], pcd_list[i], voxel=voxel)
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcd_list[j], pcd_list[i], max_correspondence_distance=voxel * 1.5, transformation=Tij
            )
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, j, Tij, info, uncertain=True))

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )
    # Extract optimized poses (world_T_i)
    transforms = [np.linalg.inv(node.pose) for node in pose_graph.nodes]
    return transforms

# file loader
def file_loader(folder):
    left_paths = sorted(folder.glob("*_left.jpg"))
    pairs = []
    for lp in left_paths:
        rp = folder / lp.name.replace("_left.jpg", "_right.jpg")
        if rp.exists():
            pairs.append((lp, rp))
    if not pairs:
        raise FileNotFoundError("No stereo pairs found matching *_left.jpg / *_right.jpg")
    return pairs

# ---------------------------
# Main batch pipeline
# ---------------------------
def run_batch(
    folder,
    q_path,
    target_short_side,
    # SGBM params
    min_disp, max_disp, block_size, uniqueness,
    speckle_win, speckle_range, block_factor,
    # Clean / sampling
    voxel_down=0.005,   # scene units; depends on your Q
    stat_nb_neighbors=20, stat_std_ratio=2.0,
    z_clip=None,        # e.g., (-0.2, 0.6) to keep reasonable depth range
    save_dir="recon_out",
    do_registration=True
):
    folder = Path(folder)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load Q
    Q = np.load(q_path)
    assert Q.shape == (4, 4), "Q must be 4x4"

    # Discover stereo pairs
    pairs = file_loader(folder)

    # Create matcher
    sgbm = make_sgbm(min_disp, max_disp, block_size, uniqueness, speckle_win, speckle_range, block_factor)

    all_pcds = []
    for i, (lp, rp) in enumerate(pairs):
        # Read
        imgL_full = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        imgR_full = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if imgL_full is None or imgR_full is None:
            print(f"Skipping {lp.name} - could not read images.")
            continue

        # Preprocess (grayscale + resize + CLAHE)
        imgL = preprocess_gray(imgL_full, target_short_side=target_short_side, clahe=True)
        imgR = preprocess_gray(imgR_full, target_short_side=target_short_side, clahe=True)

        # Compute disparity (OpenCV returns fixed-point *16)
        disp = sgbm.compute(imgL, imgR).astype(np.float32) / 16.0

        # Build a validity mask (disparity > min_disp and finite)
        valid_mask = disp > float(min_disp)

        # visualise
        try:
            viz, _ = visualize_disparity(disp, imgL, valid_mask=valid_mask, show=False)
            if save_dir is not None:
                viz.save(str(save_dir / f"{lp.stem.replace('_left','')}_disp.png"))
        except Exception:
            pass
        

        # Reproject to 3D
        pts, cols = disparity_to_points(disp, imgL, Q, valid_mask, z_clip=z_clip, flip_z=False)
        
        # Convert to Open3D point cloud and clean
        pcd = to_o3d(pts, cols)
        pcd = downsample_and_denoise(pcd, voxel=voxel_down,
                                    nb_neighbors=stat_nb_neighbors,
                                    std_ratio=stat_std_ratio)
        # Save per-view cloud
        if save_dir is not None:
            o3d.io.write_point_cloud(
                str(save_dir / f"{lp.stem.replace('_left','')}.ply"),
                pcd,
                write_ascii=True,
                print_progress=False
            )
        all_pcds.append(pcd)

    fused = None
    if do_registration and len(all_pcds) >= 2:
        print("Registering point clouds...")
        # Global multiway registration
        transforms = multiway_register(all_pcds, voxel=max(voxel_down, 0.01))

        # Transform and merge
        transformed = []
        for pcd, T in zip(all_pcds, transforms):
            p = o3d.geometry.PointCloud(pcd)  # copy
            p.transform(T)
            transformed.append(p)

        fused = transformed[0]
        for p in transformed[1:]:
            fused += p
        fused = fused.voxel_down_sample(voxel_size=voxel_down)
        fused, _ = fused.remove_statistical_outlier(nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio)
        o3d.io.write_point_cloud(str(save_dir / "fused_registered.ply"), fused)
        print(f"Saved fused cloud to {save_dir / 'fused_registered.ply'}")

        # Quick interactive 3D view (Open3D viewer)
        try:
            o3d.visualization.draw_geometries([fused])
        except Exception:
            # Fallback to Plotly if the native viewer isn't available
            pts = np.asarray(fused.points)
            cols = np.asarray(fused.colors) if fused.has_colors() else None
            plot_points_plotly(pts, cols, title="Fused Registered Cloud")

    return {"pairs": len(pairs), "saved_dir": str(save_dir), "fused": fused is not None}

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    folder = r"C:\\Github\\FSR_DLNR\\face_laser_projector\\no_markers-09_11"
    q_path = r"C:\\Github\\FSR_DLNR\\face_laser_projector\\disp_to_depth_mat.npy"

      # TUNE these to your scene scale (units come from Q). Start conservative:
    results = run_batch(
        folder            = folder,
        save_dir          = "recon_out",   # where to save
        q_path            = q_path,
        target_short_side = 720,           # resie image size
        min_disp          = -250,          # miniumum disparity
        max_disp          = -50,           # ensure (max-min) divisible by 16 (here: 104 -> rounded to 112)
        block_size        = 5,             # for sgbm
        uniqueness        = 8,             # sgbm uniquness filtering
        speckle_win       = 500,           # sgbm speckle filtering
        speckle_range     = 2,             # sgbm speckle filtering
        block_factor      = 1,             # 1 works best 
        voxel_down        = 0.5,           # subsample voxel size in mm
        stat_nb_neighbors = 24,            # number of neibours for normal computation
        stat_std_ratio    = 2.0,           # point removal agressiveness. lower = more agressive
        z_clip            = None,          # clip Z values (H\L) to clear noise 
        do_registration   = True
    )
    print(results)
