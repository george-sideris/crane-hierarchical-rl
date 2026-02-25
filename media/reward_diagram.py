"""Generate reward computation diagram for the paper.

Shows:
1. Alignment: dot product between grapple Y-axis and log Y-axis (length axis)
2. Stability: dot product between grapple Z-axis (up) and world Z-axis (up)
3. Reward formula breakdown
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
import numpy as np

fig = plt.figure(figsize=(14, 7))

# ============================================================
# LEFT PANEL: Alignment (top-down view)
# ============================================================
ax1 = fig.add_axes([0.03, 0.08, 0.44, 0.82])
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.0, 3.5)
ax1.set_aspect('equal')
ax1.set_title('(a) Alignment Score — Top-Down View', fontsize=13, fontweight='bold', pad=12)
ax1.axis('off')

# Draw grapple (rectangle, slightly rotated)
grapple_angle = 15  # degrees
g_rad = np.radians(grapple_angle)
grapple_cx, grapple_cy = 0.0, 0.5
grapple_w, grapple_h = 3.0, 0.8

# Grapple body
grapple_rect = mpatches.FancyBboxPatch(
    (-grapple_w/2, -grapple_h/2), grapple_w, grapple_h,
    boxstyle="round,pad=0.05",
    facecolor='#4a90d9', edgecolor='#2c3e6b', linewidth=2, alpha=0.7,
    transform=plt.matplotlib.transforms.Affine2D().rotate_deg(grapple_angle).translate(grapple_cx, grapple_cy) + ax1.transData
)
ax1.add_patch(grapple_rect)
ax1.text(grapple_cx, grapple_cy + 0.0, 'GRAPPLE', fontsize=9, ha='center', va='center',
         fontweight='bold', color='white', rotation=grapple_angle)

# Grapple Y-axis arrow (along the length of the grapple)
arrow_len = 2.5
gx_end = grapple_cx + arrow_len * np.cos(g_rad)
gy_end = grapple_cy + arrow_len * np.sin(g_rad)
ax1.annotate('', xy=(gx_end, gy_end), xytext=(grapple_cx, grapple_cy),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))
ax1.text(gx_end + 0.15, gy_end + 0.15, r'$\hat{y}_{grapple}$', fontsize=13, color='#e74c3c', fontweight='bold')

# Draw logs (cylinders seen from above = rectangles)
log_colors = ['#8B6914', '#7a5c12', '#9b7924']
log_angles_deg = [5, -8, 12]  # slight variation
log_positions = [(-0.3, -1.0), (0.1, -1.6), (-0.5, -2.2)]

for i, (lx, ly) in enumerate(log_positions):
    la_deg = log_angles_deg[i]
    la_rad = np.radians(la_deg)
    log_w, log_h = 2.4, 0.35
    log_rect = mpatches.FancyBboxPatch(
        (-log_w/2, -log_h/2), log_w, log_h,
        boxstyle="round,pad=0.03",
        facecolor=log_colors[i], edgecolor='#4a3008', linewidth=1.5,
        transform=plt.matplotlib.transforms.Affine2D().rotate_deg(la_deg).translate(lx, ly) + ax1.transData
    )
    ax1.add_patch(log_rect)

    # Log Y-axis arrow (along the length of the log)
    log_arrow_len = 1.8
    lx_end = lx + log_arrow_len * np.cos(la_rad)
    ly_end = ly + log_arrow_len * np.sin(la_rad)
    ax1.annotate('', xy=(lx_end, ly_end), xytext=(lx, ly),
                 arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.0))
    if i == 0:
        ax1.text(lx_end + 0.15, ly_end + 0.1, r'$\hat{y}_{log}$', fontsize=13, color='#27ae60', fontweight='bold')

# Draw angle arc between grapple Y and first log Y
# Angles are now along the length (no +90 offset)
la0 = log_angles_deg[0]
arc_center = (log_positions[0][0], log_positions[0][1])
angle_start = la0  # log length direction
angle_end = grapple_angle  # grapple length direction
arc = Arc(arc_center, 1.6, 1.6, angle=0, theta1=min(angle_start, angle_end),
          theta2=max(angle_start, angle_end), color='#e67e22', lw=2, linestyle='--')
ax1.add_patch(arc)
ax1.text(arc_center[0] + 0.9, arc_center[1] + 0.15, r'$\theta$', fontsize=14, color='#e67e22', fontweight='bold')

# Formula box for alignment
formula_text = (
    r'$\mathrm{alignment} = |\hat{y}_{grapple} \cdot \hat{y}_{log}|^{\,8}$'
    '\n'
    r'$= |\cos\theta|^{\,8}$'
)
ax1.text(0.0, -2.9, formula_text, fontsize=12, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffeaa7', edgecolor='#d4ac0d', alpha=0.9))

# ============================================================
# RIGHT PANEL: Stability (side view)
# ============================================================
ax2 = fig.add_axes([0.52, 0.08, 0.44, 0.82])
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-1.5, 5.0)
ax2.set_aspect('equal')
ax2.set_title('(b) Stability Score — Side View', fontsize=13, fontweight='bold', pad=12)
ax2.axis('off')

# Draw crane arm (simplified)
ax2.plot([0, 0], [4.5, 2.5], color='#7f8c8d', lw=4, solid_capstyle='round')
ax2.plot([0, 0.8], [2.5, 1.8], color='#7f8c8d', lw=3, solid_capstyle='round')

# Tilted grapple
tilt_angle = -18  # degrees tilt
t_rad = np.radians(tilt_angle)
grapple_cx2, grapple_cy2 = 0.8, 1.2
grapple_w2, grapple_h2 = 2.0, 0.5

grapple_rect2 = mpatches.FancyBboxPatch(
    (-grapple_w2/2, -grapple_h2/2), grapple_w2, grapple_h2,
    boxstyle="round,pad=0.05",
    facecolor='#4a90d9', edgecolor='#2c3e6b', linewidth=2, alpha=0.7,
    transform=plt.matplotlib.transforms.Affine2D().rotate_deg(tilt_angle).translate(grapple_cx2, grapple_cy2) + ax2.transData
)
ax2.add_patch(grapple_rect2)

# Grapple fingers (two lines hanging down, tilted)
finger_len = 0.9
for side in [-0.7, 0.7]:
    fx = grapple_cx2 + side * np.cos(t_rad)
    fy = grapple_cy2 + side * np.sin(t_rad)
    fx_end = fx - finger_len * np.sin(t_rad)
    fy_end = fy - finger_len * np.cos(t_rad)
    ax2.plot([fx, fx_end], [fy, fy_end], color='#2c3e6b', lw=2.5)

# Logs in grapple (circles = cross section)
log_cross_positions = [
    (grapple_cx2 - 0.4, grapple_cy2 - 0.7),
    (grapple_cx2 + 0.3, grapple_cy2 - 0.6),
    (grapple_cx2 - 0.1, grapple_cy2 - 1.1),
    (grapple_cx2 + 0.5, grapple_cy2 - 1.0),
]
for (lx, ly) in log_cross_positions:
    circle = plt.Circle((lx, ly), 0.18, facecolor='#8B6914', edgecolor='#4a3008', lw=1.5)
    ax2.add_patch(circle)

# Grapple Z-axis (up) - tilted
z_arrow_len = 2.0
zx_end = grapple_cx2 + z_arrow_len * np.sin(-t_rad)
zy_end = grapple_cy2 + z_arrow_len * np.cos(-t_rad)
ax2.annotate('', xy=(zx_end, zy_end), xytext=(grapple_cx2, grapple_cy2),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))
ax2.text(zx_end + 0.15, zy_end + 0.1, r'$\hat{z}_{grapple}$', fontsize=13, color='#e74c3c', fontweight='bold')

# World Z-axis (up) - vertical
world_z_len = 2.0
ax2.annotate('', xy=(grapple_cx2, grapple_cy2 + world_z_len),
             xytext=(grapple_cx2, grapple_cy2),
             arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2.5, linestyle='--'))
ax2.text(grapple_cx2 + 0.15, grapple_cy2 + world_z_len + 0.15, r'$\hat{z}_{world}$', fontsize=13, color='#2980b9', fontweight='bold')

# Tilt angle arc
arc2 = Arc((grapple_cx2, grapple_cy2), 2.2, 2.2, angle=90,
           theta1=0, theta2=abs(tilt_angle), color='#e67e22', lw=2, linestyle='--')
ax2.add_patch(arc2)
ax2.text(grapple_cx2 + 0.55, grapple_cy2 + 1.5, r'$\phi$', fontsize=14, color='#e67e22', fontweight='bold')

# Ground line
ax2.plot([-3.5, 3.5], [-1.0, -1.0], color='#95a5a6', lw=1.5, linestyle='-')
ax2.fill_between([-3.5, 3.5], [-1.5, -1.5], [-1.0, -1.0], color='#ecf0f1', alpha=0.5)

# Formula box for stability
formula_text2 = (
    r'$\mathrm{stability} = (\hat{z}_{grapple} \cdot \hat{z}_{world})^{\,4}$'
    '\n'
    r'$= (\cos\phi)^{\,4}$'
)
ax2.text(0.0, -0.5, formula_text2, fontsize=12, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', edgecolor='#27ae60', alpha=0.9))

# ============================================================
# Bottom: Full reward formula
# ============================================================
fig.text(0.50, 0.01,
         r'$R = n_{grasped} \;\times\; \mathrm{alignment} \;\times\; \mathrm{stability}$'
         r'$\qquad$'
         r'($R = -1$ if $n_{grasped} = 0$)',
         fontsize=14, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fadbd8', edgecolor='#e74c3c', alpha=0.9))

plt.savefig('/home/george/IsaacLab/crane_testbed/media/reward_diagram.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/home/george/IsaacLab/crane_testbed/media/reward_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: reward_diagram.png and reward_diagram.pdf")
plt.show()
