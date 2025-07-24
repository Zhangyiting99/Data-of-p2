import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import acos, degrees

# 参数设置
cutoff_dist = 20.0
bond_cutoff = 2.0
angle_center_atom_type = 7
oxygen_types = {3, 4, 5, 8}
si_type = 2
al_type = 7

files = {
    '0ps': 'dump.0.ind.lammpstrj',
    '30ps': 'dump.300000.ind.lammpstrj',
    '60ps': 'dump.600000.ind.lammpstrj'
}

def read_dump(filename):
    with open(filename) as f:
        lines = f.readlines()
    atoms = []
    for i, line in enumerate(lines):
        if "ITEM: ATOMS" in line:
            headers = line.strip().split()[2:]
            for atom_line in lines[i+1:]:
                atoms.append(atom_line.strip().split())
            break
    df = pd.DataFrame(atoms, columns=headers).astype(float)
    return df

tip_df_30ps = read_dump(files["30ps"])
tip_coord = np.array([29.8, 19.3, 52.0])

def get_neighbors(df, center_id, center_type, cutoff):
    center_atom = df[df["id"] == center_id].iloc[0]
    center_pos = np.array([center_atom["x"], center_atom["y"], center_atom["z"]])
    df["dist"] = np.linalg.norm(df[["x", "y", "z"]].values - center_pos, axis=1)
    neighbors = df[(df["dist"] < cutoff) & (df["id"] != center_id)]
    return neighbors

def is_bridging_o(df, o_id, si_al_ids):
    o_pos = df.loc[df['id'] == o_id, ['x', 'y', 'z']].values[0]
    connected = []
    for atom_id in si_al_ids:
        a_pos = df.loc[df['id'] == atom_id, ['x', 'y', 'z']].values[0]
        if np.linalg.norm(a_pos - o_pos) < bond_cutoff:
            connected.append(atom_id)
    return connected if len(connected) == 2 else []

def analyze_tetrahedra(df):
    near_tip = df[np.linalg.norm(df[["x", "y", "z"]].values - tip_coord, axis=1) < cutoff_dist]
    al_atoms = near_tip[near_tip["type"] == al_type]

    results = {"Q2": [], "Q3": []}
    si_al_df = df[df["type"].isin([al_type, si_type])]

    for _, al in al_atoms.iterrows():
        al_id = al["id"]
        neighbors = get_neighbors(df, al_id, al_type, bond_cutoff + 0.5)
        o_neighbors = neighbors[neighbors["type"].isin(oxygen_types)]
        o_ids = o_neighbors["id"].tolist()

        bridging_count = 0
        bridging_o = []
        for oid in o_ids:
            bridging = is_bridging_o(df, oid, si_al_df["id"].tolist())
            if bridging:
                bridging_count += 1
                bridging_o.append(oid)

        if bridging_count == 2:
            qtype = "Q2"
        elif bridging_count == 3:
            qtype = "Q3"
        else:
            continue

        bond_length = []
        for oid in o_ids:
            o_pos = df[df["id"] == oid][["x", "y", "z"]].values[0]
            al_pos = al[["x", "y", "z"]].values
            bond_length.append(np.linalg.norm(o_pos - al_pos))

        bond_vector = []
        for oid in o_ids:
            o_pos = df[df["id"] == oid][["x", "y", "z"]].values[0]
            bond_vector.append(o_pos - al[["x", "y", "z"]].values)

        bond_angle = []
        for i in range(len(bond_vector)):
            for j in range(i+1, len(bond_vector)):
                v1 = bond_vector[i]
                v2 = bond_vector[j]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = degrees(acos(cos_angle))
                bond_angle.append(angle)

        avg_bond_length = np.mean(bond_length) if bond_length else 0
        avg_bond_angle = np.mean(bond_angle) if bond_angle else 0

        results[qtype].append({
            "id": al_id,
            "bond_length": bond_length,
            "bond_angle": bond_angle,
            "avg_bond_length": avg_bond_length,
            "avg_bond_angle": avg_bond_angle
        })

    return results

# 主分析流程
all_data = {}
for time, file in files.items():
    df = read_dump(file)
    result = analyze_tetrahedra(df)
    all_data[time] = result

    for qtype in ["Q2", "Q3"]:
        rows_bl = []
        rows_ba = []
        for entry in result[qtype]:
            for bl in entry["bond_length"]:
                rows_bl.append({"atom_id": entry["id"], "bond_length": bl})
            for ba in entry["bond_angle"]:
                rows_ba.append({"atom_id": entry["id"], "bond_angle": ba})

        pd.DataFrame(rows_bl).to_csv(f"{qtype}_bond_length_{time}.csv", index=False)
        pd.DataFrame(rows_ba).to_csv(f"{qtype}_bond_angle_{time}.csv", index=False)

        # 输出平均统计
        if result[qtype]:
            avg_bl = np.mean([entry["avg_bond_length"] for entry in result[qtype]])
            avg_ba = np.mean([entry["avg_bond_angle"] for entry in result[qtype]])
            print(f"{time} - {qtype}: Avg Bond Length = {avg_bl:.4f} Å, Avg Bond Angle = {avg_ba:.2f}°")
        else:
            print(f"{time} - {qtype}: No data")

# 计算键长、键角变化 Δ
def compute_deltas(qtype, param, time1, time2):
    df1 = pd.read_csv(f"{qtype}_{param}_{time1}.csv")
    df2 = pd.read_csv(f"{qtype}_{param}_{time2}.csv")

    common_ids = set(df1["atom_id"]).intersection(df2["atom_id"])
    deltas = []
    for aid in common_ids:
        v1 = df1[df1["atom_id"] == aid][param].values
        v2 = df2[df2["atom_id"] == aid][param].values
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        diff = v2 - v1
        for d in diff:
            deltas.append(d)

    delta_df = pd.DataFrame({f"{param}_delta": deltas})
    delta_df.to_csv(f"{qtype}_{param}_delta_{time1}_{time2}.csv", index=False)

    mean_abs = np.mean(np.abs(deltas))
    std_abs = np.std(np.abs(deltas))

    print(f"{qtype} {param} Δ Mean Abs ({time1} to {time2}): {mean_abs:.4f}, Std: {std_abs:.4f}")

# 批量计算Δ（仅计算 0ps 到 30ps 和 0ps 到 60ps 的变化）
for q in ["Q2", "Q3"]:
    for p in ["bond_length", "bond_angle"]:
        compute_deltas(q, p, time1="0ps", time2="30ps")
        compute_deltas(q, p, time1="0ps", time2="60ps")
