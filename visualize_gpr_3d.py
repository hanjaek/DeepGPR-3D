from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# threshold for bright voxels
THRESH = 0.7
mask = volume > THRESH
idx  = np.argwhere(mask)  # (N, 3)

if idx.size == 0:
    print("No voxels above threshold. Try lower THRESH.")
else:
    s_idx = idx[:, 0]
    z_idx = idx[:, 1]
    y_idx = idx[:, 2]

    dx = meta["spacing"]["dx_m"]
    dy = meta["spacing"]["dy_m"]
    dz = meta["spacing"]["dz_m"]

    x = s_idx * dx
    y = y_idx * dy
    z = z_idx * dz

    intens = volume[mask].ravel()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # small subset if too many points
    max_points = 50000
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x = x[indices]
        y = y[indices]
        z = z[indices]
        intens = intens[indices]

    p = ax.scatter(x, y, z, c=intens, s=1)
    ax.set_xlabel("X (slice direction)")
    ax.set_ylabel("Y (along-track)")
    ax.set_zlabel("Z (depth)")
    ax.set_title("3D GPR Volume (thresholded points)")

    plt.show()
