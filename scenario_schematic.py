# -*- coding: utf-8 -*-
'''UAV-BS 布局场景示意图'''
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from mpl_toolkits.mplot3d import proj3d

# 中文字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(11)

fig = plt.figure(figsize=(14, 9), dpi=140)
ax = fig.add_subplot(111, projection='3d')

# ---------------- ground: 500 x 500 m ----------------
B = 250
xx, yy = np.meshgrid(np.linspace(-B, B, 2), np.linspace(-B, B, 2))
ax.plot_surface(xx, yy, np.zeros_like(xx), color='#e8eef5', alpha=0.55, linewidth=0)
# boundary
for (x0, y0), (x1, y1) in [((-B, -B), (B, -B)), ((B, -B), (B, B)),
                            ((B, B), (-B, B)), ((-B, B), (-B, -B))]:
    ax.plot([x0, x1], [y0, y1], [0, 0], color='#555', lw=1.2)
# grid
for g in range(-B, B + 1, 50):
    ax.plot([g, g], [-B, B], [0, 0], color='#bbb', lw=0.3, alpha=0.6)
    ax.plot([-B, B], [g, g], [0, 0], color='#bbb', lw=0.3, alpha=0.6)

# ---------------- 4 K-means zones (quadrants) ----------------
zone_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# draw dashed zone borders
ax.plot([0, 0], [-B, B], [0, 0], color='#888', ls='--', lw=1)
ax.plot([-B, B], [0, 0], [0, 0], color='#888', ls='--', lw=1)

# ---------------- users (3 traffic types) ----------------
# type 0 Web, type 1 Video, type 2 Messaging
types = np.random.choice([0, 1, 2], size=80, p=[0.4, 0.35, 0.25])
ux = np.random.uniform(-B + 15, B - 15, 80)
uy = np.random.uniform(-B + 15, B - 15, 80)
tcolor = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
tmarker = {0: 'o', 1: 's', 2: '^'}
tname = {0: '网页 2 Mbps', 1: '视频 5 Mbps', 2: '消息 0.1 Mbps'}
for i in range(80):
    ax.scatter(ux[i], uy[i], 0, c=tcolor[types[i]], marker=tmarker[types[i]],
               s=18, depthshade=False, alpha=0.85)

# ---------------- airship (fixed aerial hub) ----------------
ax.scatter([0], [0], [250], c='#7b1fa2', marker='*', s=400,
           depthshade=False, edgecolors='k', linewidths=0.6, zorder=10)
ax.text(0, 0, 265, '飞艇基站（0,0,250 米）\n回程中枢  每架无人机 5 MHz',
        ha='center', fontsize=9, fontweight='bold', color='#4a148c')

# ---------------- 4 relay drones ----------------
drone_pos = [(-140, 130, 175), (-120, -140, 150),
             (135, -125, 160), (145, 140, 180)]
drone_lbl = ['中继无人机 1', '中继无人机 2', '中继无人机 3', '中继无人机 4']
zmin, zmax = 100, 200

# altitude axis marker
ax.plot([B + 30, B + 30], [B + 30, B + 30], [zmin, zmax], color='#333', lw=1.5)
ax.text(B + 35, B + 30, (zmin + zmax) / 2, f'高度 {zmin}-{zmax} 米',
        fontsize=8, color='#333', va='center')

for i, (dx, dy, dz) in enumerate(drone_pos):
    c = zone_colors[i]
    # backhaul link airship -> drone
    ax.plot([0, dx], [0, dy], [250, dz], color='#7b1fa2', ls='-', lw=1.8, alpha=0.85)
    # drone glyph
    ax.scatter([dx], [dy], [dz], c=c, marker='^', s=160,
               depthshade=False, edgecolors='k', linewidths=0.7, zorder=9)
    ax.text(dx, dy, dz + 14, drone_lbl[i], ha='center', fontsize=8.5,
            fontweight='bold', color=c)
    # altitude drop line
    ax.plot([dx, dx], [dy, dy], [0, dz], color=c, ls=':', lw=1, alpha=0.7)

    # access links to ~5 nearby users of that zone
    # find users in this zone
    qx = ux[(np.sign(ux) == np.sign(dx)) & (np.sign(uy) == np.sign(dy))]
    qy = uy[(np.sign(ux) == np.sign(dx)) & (np.sign(uy) == np.sign(dy))]
    idxs = np.where((np.sign(ux) == np.sign(dx)) & (np.sign(uy) == np.sign(dy)))[0]
    # pick 5 closest
    if len(idxs) > 0:
        d2 = (ux[idxs] - dx) ** 2 + (uy[idxs] - dy) ** 2
        pick = idxs[np.argsort(d2)[:5]]
        for j in pick:
            ax.plot([dx, ux[j]], [dy, uy[j]], [dz, 0], color=c, lw=0.7, alpha=0.5)

# ---------------- legends (manual) ----------------
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#7b1fa2',
           markersize=14, label='飞艇基站（回程中枢）'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='tab:blue',
           markersize=10, label='中继无人机基站（接入，3D 移动）'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=tcolor[0],
           markersize=7, label='网页用户（2 Mbps）'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=tcolor[1],
           markersize=7, label='视频用户（5 Mbps）'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=tcolor[2],
           markersize=7, label='消息用户（0.1 Mbps）'),
    Line2D([0], [0], color='#7b1fa2', lw=2, label='回程链路（飞艇→无人机）'),
    Line2D([0], [0], color='#888', lw=1, ls='--', label='K 均值聚类边界'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
          framealpha=0.9, borderpad=0.8)

# ---------------- title & labels ----------------
ax.set_title('本文场景：飞艇回程的中继无人机基站\n'
             '基于 Lyapunov + REINFORCE 的自适应 3D 布局',
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('X（米）', fontsize=9)
ax.set_ylabel('Y（米）', fontsize=9)
ax.set_zlabel('高度 z（米）', fontsize=9)
ax.set_xlim(-B - 20, B + 20)
ax.set_ylim(-B - 20, B + 20)
ax.set_zlim(0, 300)
ax.view_init(elev=28, azim=-58)

# annotation box
note = ('每时隙流程：\n'
        '  1) K 均值对用户聚类\n'
        '  2) 策略网络提出 K=20 个候选动作\n'
        '  3) Lyapunov 打分 J = -(队列漂移+虚拟队列漂移)+V·MOS\n'
        '  4) 执行最优候选，二分法求交付速率\n'
        '  5) REINFORCE 更新策略')
ax.text2D(0.02, 0.02, note, transform=ax.transAxes, fontsize=7.5,
          va='bottom', ha='left',
          bbox=dict(boxstyle='round,pad=0.4', fc='#fffde7', ec='#fbc02d', alpha=0.95))

plt.tight_layout()
plt.savefig('scenario_schematic.png', dpi=140, bbox_inches='tight',
            facecolor='white')
plt.close()
print('saved scenario_schematic.png')
