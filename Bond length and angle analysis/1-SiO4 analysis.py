import numpy as np
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ------- 配置参数 --------
bond_cutoff = 2.0  # Si-O最大键长，单位Å，根据实际调整
probe_center = np.array([29.8, 19.3, 52.0])  # 替换成压头中心坐标，单位Å
probe_radius = 20.0  # 压头附近判定半径，单位Å，根据实际调整

# 原子类型定义
calcium_type = 1
silicon_type = 2
aluminum_type = 7
oxygen_types = [3, 4, 5, 8]
hydrogen_types = [6, 9]

# dump文件夹和文件名模板
dump_folder = "./"  # dump文件所在文件夹
dump_pattern = "dump.*.ind.lammpstrj"


def read_dump(filename):
    """读取单个dump文件，返回atoms列表，格式为{'id': int, 'type': int, 'pos': np.array}"""
    atoms = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    natoms = 0
    start = 0
    for i, line in enumerate(lines):
        if "ITEM: NUMBER OF ATOMS" in line:
            natoms = int(lines[i + 1].strip())
        if "ITEM: ATOMS id type x y z" in line:
            start = i + 1
            break
    for line in lines[start:start + natoms]:
        parts = line.strip().split()
        aid = int(parts[0])
        atype = int(parts[1])
        pos = np.array(list(map(float, parts[2:5])))
        atoms.append({'id': aid, 'type': atype, 'pos': pos})
    return atoms


def dist(a, b):
    return np.linalg.norm(a - b)


def find_bridging_oxygens(atoms):
    """找桥氧：与两个或以上 Si/Al 原子相连的氧原子 id 集合"""
    atom_dict = {a['id']: a for a in atoms}
    network_former_ids = [a['id'] for a in atoms if a['type'] in (silicon_type, aluminum_type)]

    bridging_ox = set()

    for o in [a for a in atoms if a['type'] in oxygen_types]:
        o_pos = o['pos']
        connected_formers = 0
        for nid in network_former_ids:
            if dist(atom_dict[nid]['pos'], o_pos) < bond_cutoff:
                connected_formers += 1
                if connected_formers >= 2:
                    bridging_ox.add(o['id'])
                    break

    return bridging_ox


def identify_Qn(atoms, bridging_ox):
    """
    根据桥氧数判断Si的Qn类型
    返回：Q2列表，Q3列表，Si-O连接字典{Si_id: [O_id,...]}
    """
    atom_dict = {a['id']: a for a in atoms}
    si_ids = [a['id'] for a in atoms if a['type'] == silicon_type]
    oxygen_atoms = [a for a in atoms if a['type'] in oxygen_types]

    si_o_neighbors = {}
    for si in si_ids:
        si_pos = atom_dict[si]['pos']
        connected_ox = []
        for o in oxygen_atoms:
            if dist(si_pos, o['pos']) < bond_cutoff:
                connected_ox.append(o['id'])
        si_o_neighbors[si] = connected_ox

    Q2 = []
    Q3 = []
    for si, o_list in si_o_neighbors.items():
        bridge_count = sum(1 for o in o_list if o in bridging_ox)
        if bridge_count == 2:
            Q2.append(si)
        elif bridge_count == 3:
            Q3.append(si)
    return Q2, Q3, si_o_neighbors


def compute_tetrahedron_geometry(si_id, o_ids, atom_dict):
    """
    计算Si-O四面体几何参数：4个键长，6个O-Si-O键角
    """
    si_pos = atom_dict[si_id]['pos']
    o_pos_list = [atom_dict[oid]['pos'] for oid in o_ids]
    bonds = [dist(si_pos, o_pos) for o_pos in o_pos_list]

    angles = []
    for i in range(4):
        for j in range(i + 1, 4):
            vec1 = o_pos_list[i] - si_pos
            vec2 = o_pos_list[j] - si_pos
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = min(1.0, max(-1.0, cos_angle))
            angle_deg = np.degrees(np.arccos(cos_angle))
            angles.append(angle_deg)
    return bonds, angles


def is_near_probe(pos):
    return np.linalg.norm(pos - probe_center) <= probe_radius


def make_atom_dict(atoms):
    return {a['id']: a for a in atoms}


def get_tetra_info(si_ids, atoms, si_o_neighbors):
    """计算指定Si列表的四面体几何信息"""
    atom_dict = make_atom_dict(atoms)
    tetra_info = {}
    for si in si_ids:
        oids = si_o_neighbors.get(si, [])[:4]
        if len(oids) < 4:
            continue
        bonds, angles = compute_tetrahedron_geometry(si, oids, atom_dict)
        tetra_info[si] = {'bonds': bonds, 'angles': angles, 'o_ids': oids}
    return tetra_info


def extract_data(file_path, qtype):
    """提取数据并计算绝对值变化"""
    df = pd.read_csv(file_path)
    df_q = df[df['Q_type'] == qtype]

    # 计算绝对值变化
    bond_cols = ['bond1_delta', 'bond2_delta', 'bond3_delta', 'bond4_delta']
    angle_cols = ['angle1_delta', 'angle2_delta', 'angle3_delta',
                  'angle4_delta', 'angle5_delta', 'angle6_delta']

    bonds = df_q[bond_cols].abs().values.flatten()
    angles = df_q[angle_cols].abs().values.flatten()

    bonds = bonds[~np.isnan(bonds)]
    angles = angles[~np.isnan(angles)]
    return bonds, angles


def plot_kde_comparison(file_dict, qtype, value_type, save_folder, bw_adjust=0.8):
    plt.figure(figsize=(6, 5))

    # 设置横坐标标签和范围
    if value_type == 'bond':
        xlim = (0, 0.2)  # 绝对值变化范围调整为0开始
        xlabel = "Absolute Bond Length Change (Å)"
    else:
        xlim = (0, 20)  # 绝对值变化范围调整为0开始
        xlabel = "Absolute Bond Angle Change (°)"

    for label, path in file_dict.items():
        if label == '0 ps':
            continue  # 跳过0ps，因为它是基准

        bonds, angles = extract_data(path, qtype)
        data = bonds if value_type == 'bond' else angles
        if len(data) > 0:
            sns.kdeplot(data, label=label, bw_adjust=bw_adjust, linewidth=2)

    plt.xlim(xlim)

    # 设置字体大小 + 字体为 Arial
    plt.xlabel(xlabel, fontsize=24, fontname='Arial')
    plt.ylabel("Density", fontsize=24, fontname='Arial')

    plt.legend(fontsize=20, title_fontsize=20)
    plt.xticks(fontsize=18, fontname='Arial')
    plt.yticks(fontsize=18, fontname='Arial')

    # 设置标尺刻度线显示
    plt.tick_params(axis='both', which='both', length=4, width=1, direction='in', color='black')
    plt.subplots_adjust(bottom=0.12, left=0.12, right=0.95, top=0.95)
    plt.tight_layout()

    # 设置四边边框颜色为黑色
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)

    os.makedirs(save_folder, exist_ok=True)
    save_name = f"{qtype}_{value_type}_abs_delta_kde.png"
    plt.savefig(os.path.join(save_folder, save_name), dpi=300)
    plt.close()
    print(f"Saved: {save_name}")


def calculate_statistics(file_dict, qtype):
    """计算绝对值变化的统计量"""
    stats = []

    for label, path in file_dict.items():
        if label == '0 ps':
            continue  # 跳过基准

        bonds, angles = extract_data(path, qtype)

        if len(bonds) > 0:
            bond_mean = np.mean(bonds)
            bond_std = np.std(bonds)
            bond_median = np.median(bonds)
            bond_max = np.max(bonds)
        else:
            bond_mean = bond_std = bond_median = bond_max = np.nan

        if len(angles) > 0:
            angle_mean = np.mean(angles)
            angle_std = np.std(angles)
            angle_median = np.median(angles)
            angle_max = np.max(angles)
        else:
            angle_mean = angle_std = angle_median = angle_max = np.nan

        stats.append({
            'time': label,
            'Q_type': qtype,
            'bond_mean_abs': bond_mean,
            'bond_std_abs': bond_std,
            'bond_median_abs': bond_median,
            'bond_max_abs': bond_max,
            'angle_mean_abs': angle_mean,
            'angle_std_abs': angle_std,
            'angle_median_abs': angle_median,
            'angle_max_abs': angle_max
        })

    return pd.DataFrame(stats)


def main():
    # 读取所有dump文件，按时间戳排序，只选择0ps、30ps和60ps
    dump_files = sorted(glob.glob(os.path.join(dump_folder, dump_pattern)),
                        key=lambda f: int(os.path.basename(f).split('.')[1]))

    # 只保留0ps、30ps和60ps
    selected_times = [0, 300000, 600000]
    dump_files = [f for f in dump_files if int(os.path.basename(f).split('.')[1]) in selected_times]

    print("读取30ps文件判定压头附近原子...")
    atoms_30ps = read_dump(os.path.join(dump_folder, "dump.300000.ind.lammpstrj"))
    bridging_ox_30ps = find_bridging_oxygens(atoms_30ps)
    Q2_30ps, Q3_30ps, si_o_neighbors_30ps = identify_Qn(atoms_30ps, bridging_ox_30ps)

    Q2_near = [si for si in Q2_30ps if is_near_probe([a['pos'] for a in atoms_30ps if a['id'] == si][0])]
    Q3_near = [si for si in Q3_30ps if is_near_probe([a['pos'] for a in atoms_30ps if a['id'] == si][0])]
    print(f"压头附近Q2数量: {len(Q2_near)}, Q3数量: {len(Q3_near)}")

    # 读取所有时间帧数据
    all_atoms = {}
    frame_times = []
    for file in dump_files:
        time = int(os.path.basename(file).split('.')[1])
        all_atoms[time] = read_dump(file)
        frame_times.append(time)
    frame_times.sort()

    print("读取0ps基准帧，计算基准键长和键角...")
    atoms_0ps = all_atoms[0]
    bridging_ox_0ps = find_bridging_oxygens(atoms_0ps)
    Q2_0ps, Q3_0ps, si_o_neighbors_0ps = identify_Qn(atoms_0ps, bridging_ox_0ps)

    # 保证基准Q2/Q3为30ps附近的筛选子集（保证跟踪同一批）
    Q2_base = [sid for sid in Q2_0ps if sid in Q2_near]
    Q3_base = [sid for sid in Q3_0ps if sid in Q3_near]

    baseline_Q2 = get_tetra_info(Q2_base, atoms_0ps, si_o_neighbors_0ps)
    baseline_Q3 = get_tetra_info(Q3_base, atoms_0ps, si_o_neighbors_0ps)

    print("开始计算所有帧相对0ps的键长键角变化并保存csv...")
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)

    for time in frame_times:
        atoms = all_atoms[time]
        bridging_ox = find_bridging_oxygens(atoms)
        Q2, Q3, si_o_neighbors = identify_Qn(atoms, bridging_ox)

        current_Q2 = get_tetra_info(Q2_base, atoms, si_o_neighbors)
        current_Q3 = get_tetra_info(Q3_base, atoms, si_o_neighbors)

        records = []

        for q_type, base_data, curr_data in [('Q2', baseline_Q2, current_Q2), ('Q3', baseline_Q3, current_Q3)]:
            for si in base_data.keys():
                if si not in curr_data:
                    continue
                base = base_data[si]
                curr = curr_data[si]

                bond_delta = [c - b for c, b in zip(curr['bonds'], base['bonds'])]
                angle_delta = [c - b for c, b in zip(curr['angles'], base['angles'])]

                rec = {
                    'frame_ps': f"{time // 10000}ps",
                    'Q_type': q_type,
                    'center_Si_id': si,
                    'O1_id': base['o_ids'][0], 'O2_id': base['o_ids'][1],
                    'O3_id': base['o_ids'][2], 'O4_id': base['o_ids'][3],
                    'bond1_delta': bond_delta[0], 'bond2_delta': bond_delta[1],
                    'bond3_delta': bond_delta[2], 'bond4_delta': bond_delta[3],
                    'angle1_delta': angle_delta[0], 'angle2_delta': angle_delta[1],
                    'angle3_delta': angle_delta[2], 'angle4_delta': angle_delta[3],
                    'angle5_delta': angle_delta[4], 'angle6_delta': angle_delta[5]
                }
                records.append(rec)

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(output_folder, f"{time}_tetra_delta.csv"), index=False)

    print("计算完成，结果保存在 output 文件夹。")

    # 绘制KDE图
    print("开始绘制KDE分布图...")
    file_dict = {
        '0 ps': './output/0_tetra_delta.csv',
        '30 ps': './output/300000_tetra_delta.csv',
        '60 ps': './output/600000_tetra_delta.csv'
    }

    save_folder = './output_kde_plots'
    os.makedirs(save_folder, exist_ok=True)

    for qtype in ['Q2', 'Q3']:
        for value_type in ['bond', 'angle']:
            plot_kde_comparison(file_dict, qtype, value_type, save_folder, bw_adjust=0.8)

    # 计算统计量并保存
    print("计算统计量...")
    all_stats = []
    for qtype in ['Q2', 'Q3']:
        stats_df = calculate_statistics(file_dict, qtype)
        all_stats.append(stats_df)

    stats_combined = pd.concat(all_stats)
    stats_combined.to_csv(os.path.join(save_folder, 'abs_delta_statistics.csv'), index=False)

    # 打印统计结果
    print("\n键长和键角绝对值变化统计:")
    print(stats_combined.to_string(index=False))

    print("\n所有分析完成!")


if __name__ == "__main__":
    main()                                                                              