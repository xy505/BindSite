# import os
# import numpy as np
# import pandas as pd
# from Bio import PDB
# from rdkit import Chem
# from rdkit.Chem import rdMolTransforms
# from multiprocessing import Pool
#
# # -------------------------- 1. 配置路径和参数 --------------------------
# # 核心路径配置（需确认路径正确）
# ESMFOLD_DIR = os.path.abspath(r"D:\posebusters_esmfold\esmfold_prepared")  # 命名：5S8I_2LY_p.pdb
# PROTEIN_DIR = os.path.abspath(r"D:\posebusters_esmfold\ground_protein_prepared")  # 命名：5S8I_2LY_protein.pdb
# LIGAND_DIR = os.path.abspath(r"D:\posebusters_esmfold\ground_ligand")  # 命名：5S8I_2LY_ligand.sdf
# OUTPUT_CSV = os.path.abspath(r"D:\posebusters_esmfold\posebuster_data.csv")  # 输出CSV路径
# BINDING_SITE_DISTANCE = 5.0  # 结合位点距离阈值（Å），可调整
#
# # 氨基酸三字母→单字母映射
# AMINO_ACID_MAP = {
#     "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
#     "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
#     "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
#     "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
#     "SEC": "U", "PYL": "O"
# }
#
# # 初始化PDB解析器（静默模式，屏蔽警告）
# pdb_parser = PDB.PDBParser(QUIET=True)
#
#
# # -------------------------- 辅助函数：提取核心PDB ID --------------------------
# def extract_pdb_id(filename, folder_type):
#     """
#     根据文件夹类型提取核心PDB ID（适配不同命名规则）
#     :param filename: 文件名（如5S8I_2LY_p.pdb 或 5S8I_2LY_protein.pdb）
#     :param folder_type: 文件夹类型（esmfold/protein）
#     :return: 核心PDB ID（如5S8I_2LY）
#     """
#     if folder_type == "esmfold":
#         # esmfold_prepared：5S8I_2LY_p.pdb → 5S8I_2LY
#         return filename.replace("_p.pdb", "")
#     elif folder_type == "protein":
#         # ground_protein_prepared：5S8I_2LY_protein.pdb → 5S8I_2LY
#         return filename.replace("_protein.pdb", "")
#     else:
#         return None
#
#
# # -------------------------- 2. 核心函数：处理单个复合物 --------------------------
# def process_complex(pdb_filename):
#     """
#     处理单个蛋白质-配体复合物，返回PDB标识、序列、结合位点
#     :param pdb_filename: 蛋白质PDB文件名（如5S8I_2LY_protein.pdb）
#     :return: (pdb_id, sequence, binding_sites) 或 None（处理失败）
#     """
#     try:
#         # 1. 提取核心PDB ID（适配protein文件夹命名）
#         pdb_id = extract_pdb_id(pdb_filename, "protein")
#         print(f"Processing {pdb_id}...")
#
#         # 2. 读取蛋白质PDB文件，提取序列和原子坐标
#         protein_path = os.path.join(PROTEIN_DIR, pdb_filename)
#         structure = pdb_parser.get_structure(pdb_id, protein_path)
#
#         # 存储：序列、残基索引→原子坐标映射
#         protein_sequence = ""
#         residue_coords = {}  # key: 残基全局索引, value: 残基所有原子坐标列表
#         residue_index = 0  # 残基全局索引（从0开始）
#
#         # 遍历蛋白质结构（模型→链→残基→原子）
#         for model in structure:
#             for chain in model:
#                 for residue in chain:
#                     # 只处理标准氨基酸
#                     if residue.resname in AMINO_ACID_MAP:
#                         # 拼接蛋白质序列
#                         protein_sequence += AMINO_ACID_MAP[residue.resname]
#
#                         # 存储该残基的所有原子坐标
#                         residue_coords[residue_index] = []
#                         for atom in residue:
#                             if atom.element != "H":  # 忽略氢原子，减少计算量
#                                 residue_coords[residue_index].append(atom.get_coord())
#                         residue_index += 1
#
#         # 3. 读取配体SDF文件，提取配体原子坐标（用核心PDB ID匹配）
#         ligand_path = os.path.join(LIGAND_DIR, f"{pdb_id}_ligand.sdf")
#         if not os.path.exists(ligand_path):
#             print(f"Warning: 配体文件 {ligand_path} 不存在，跳过")
#             return None
#
#         # 解析SDF文件
#         ligand_mol = Chem.SDMolSupplier(ligand_path, removeHs=False)[0]
#         if ligand_mol is None:
#             print(f"Warning: 配体文件 {ligand_path} 解析失败，跳过")
#             return None
#
#         # 提取配体原子坐标
#         ligand_coords = []
#         conf = ligand_mol.GetConformer()
#         for atom_idx in range(ligand_mol.GetNumAtoms()):
#             if ligand_mol.GetAtomWithIdx(atom_idx).GetSymbol() != "H":  # 忽略氢原子
#                 coord = conf.GetAtomPosition(atom_idx)
#                 ligand_coords.append(np.array([coord.x, coord.y, coord.z]))
#         ligand_coords = np.array(ligand_coords)
#
#         # 4. 计算结合位点（配体周围BINDING_SITE_DISTANCE内的残基）
#         binding_residues = set()
#         for res_idx, atom_coords in residue_coords.items():
#             if not atom_coords:
#                 continue
#             # 残基所有原子的坐标数组
#             res_atom_coords = np.array(atom_coords)
#
#             # 计算残基原子与配体原子的最小距离
#             dists = np.sqrt(np.sum((res_atom_coords[:, None] - ligand_coords[None, :]) ** 2, axis=2))
#             min_dist = np.min(dists)
#
#             # 小于阈值则视为结合位点残基
#             if min_dist <= BINDING_SITE_DISTANCE:
#                 binding_residues.add(res_idx)
#
#         # 5. 整理结果
#         binding_sites = ",".join(map(str, sorted(binding_residues)))  # 转为逗号分隔的字符串
#         return (pdb_id, protein_sequence, binding_sites)
#
#     except Exception as e:
#         print(f"Error processing {pdb_filename}: {str(e)}")
#         return None
#
#
# # -------------------------- 3. 批量处理所有复合物 --------------------------
# def main():
#     # 步骤1：提取esmfold_prepared中的核心PDB ID（适配5S8I_2LY_p.pdb命名）
#     esmfold_files = [f for f in os.listdir(ESMFOLD_DIR) if f.endswith("_p.pdb")]
#     esmfold_pdb_ids = set([extract_pdb_id(f, "esmfold") for f in esmfold_files])
#
#     if not esmfold_pdb_ids:
#         print(f"Error: esmfold_prepared文件夹({ESMFOLD_DIR})内未找到任何_p.pdb文件")
#         return
#     print(f"从esmfold_prepared找到 {len(esmfold_pdb_ids)} 个核心PDB ID")
#
#     # 步骤2：提取ground_protein_prepared中的文件，并过滤出esmfold中存在的ID
#     all_protein_files = [f for f in os.listdir(PROTEIN_DIR) if f.endswith("_protein.pdb")]
#     # 过滤逻辑：文件的核心ID在esmfold_pdb_ids中才保留
#     filtered_protein_files = []
#     for f in all_protein_files:
#         pdb_id = extract_pdb_id(f, "protein")
#         if pdb_id in esmfold_pdb_ids:
#             filtered_protein_files.append(f)
#
#     if not filtered_protein_files:
#         print("Error: 未找到同时存在于ground_protein_prepared和esmfold_prepared的蛋白质（核心ID不匹配）")
#         return
#     print(f"找到 {len(filtered_protein_files)} 个核心ID匹配的蛋白质文件，开始处理...")
#
#     # 步骤3：多进程处理过滤后的文件
#     with Pool(processes=os.cpu_count()) as pool:
#         results = pool.map(process_complex, filtered_protein_files)
#
#     # 步骤4：过滤处理失败的结果
#     valid_results = [r for r in results if r is not None]
#
#     # 步骤5：生成DataFrame并最终过滤（确保只保留esmfold中的ID）
#     df = pd.DataFrame(valid_results, columns=["PDB", "Sequence", "BS"])
#     df = df[df["PDB"].isin(esmfold_pdb_ids)]  # 最终过滤
#     df = df[df["Sequence"].str.len() > 0]  # 过滤空序列
#     df = df[df["BS"].str.len() > 0]  # 过滤空结合位点
#
#     # 步骤6：保存CSV
#     df.to_csv(OUTPUT_CSV, index=False, quoting=1)  # quoting=1 确保BS列带引号
#     print(f"\n处理完成！结果已保存到 {OUTPUT_CSV}")
#     print(f"最终保留 {len(df)} 个符合条件的复合物（核心ID在esmfold_prepared中且处理成功）")
#
#
# if __name__ == "__main__":
#     main()


# import os
# import shutil
#
# # -------------------------- 配置路径 --------------------------
# # 原始文件夹路径（存放所有xxx_protein.pdb文件的目录）
# SOURCE_DIR = os.path.abspath(r"D:\posebusters_esmfold\ground_protein_prepared")
# # 是否执行移动操作（True=实际移动，False=仅预览操作，建议先预览）
# DRY_RUN = False
#
#
# # -------------------------- 核心函数 --------------------------
# def reorganize_protein_files(source_dir):
#     """
#     整理蛋白质文件目录结构：
#     1. 遍历source_dir中的xxx_protein.pdb文件
#     2. 为每个文件创建xxx子文件夹（去掉_protein.pdb后缀）
#     3. 将原文件移动到对应的子文件夹中
#     """
#     # 1. 获取所有xxx_protein.pdb文件
#     pdb_files = [f for f in os.listdir(source_dir)
#                  if f.endswith("_protein.pdb") and os.path.isfile(os.path.join(source_dir, f))]
#
#     if not pdb_files:
#         print(f"错误：在 {source_dir} 中未找到任何_protein.pdb文件")
#         return
#
#     print(f"找到 {len(pdb_files)} 个需要整理的蛋白质文件：")
#     for f in pdb_files:
#         print(f"  - {f}")
#
#     # 2. 逐个处理文件
#     success_count = 0
#     fail_count = 0
#     for filename in pdb_files:
#         try:
#             # 提取文件夹名（去掉_protein.pdb后缀）
#             folder_name = filename.replace("_protein.pdb", "")
#             # 拼接路径
#             file_path = os.path.join(source_dir, filename)
#             folder_path = os.path.join(source_dir, folder_name)
#
#             # 3. 创建子文件夹（如果不存在）
#             if not os.path.exists(folder_path):
#                 if DRY_RUN:
#                     print(f"[预览] 创建文件夹：{folder_path}")
#                 else:
#                     os.makedirs(folder_path, exist_ok=True)
#                     print(f"已创建文件夹：{folder_path}")
#             else:
#                 print(f"文件夹已存在：{folder_path}")
#
#             # 4. 移动文件到子文件夹
#             new_file_path = os.path.join(folder_path, filename)
#             if os.path.exists(new_file_path):
#                 print(f"警告：目标文件已存在，跳过 → {new_file_path}")
#                 fail_count += 1
#                 continue
#
#             if DRY_RUN:
#                 print(f"[预览] 移动文件：{file_path} → {new_file_path}")
#             else:
#                 shutil.move(file_path, new_file_path)
#                 print(f"已移动文件：{file_path} → {new_file_path}")
#             success_count += 1
#
#         except Exception as e:
#             print(f"处理文件 {filename} 失败：{str(e)}")
#             fail_count += 1
#
#     # 5. 输出统计结果
#     print("\n===== 整理完成 =====")
#     print(f"总文件数：{len(pdb_files)}")
#     print(f"成功整理：{success_count}")
#     print(f"失败/跳过：{fail_count}")
#
#
# # -------------------------- 执行主函数 --------------------------
# if __name__ == "__main__":
#     if DRY_RUN:
#         print("=== 预览模式（不会实际修改文件）===\n")
#     else:
#         print("=== 实际执行模式（会创建文件夹并移动文件）===\n")
#
#     reorganize_protein_files(SOURCE_DIR)

import os
import shutil

# -------------------------- 配置路径 --------------------------
# 原始文件夹路径（存放所有xxx_p.pdb文件的目录）
SOURCE_DIR = os.path.abspath(r"D:\posebusters_esmfold\esmfold_prepared")
# 是否执行移动操作（True=实际移动，False=仅预览操作，建议先预览）
DRY_RUN = False


# -------------------------- 核心函数 --------------------------
def reorganize_protein_files(source_dir):
    """
    整理蛋白质文件目录结构：
    1. 遍历source_dir中的xxx_p.pdb文件
    2. 为每个文件创建xxx子文件夹（去掉_p.pdb后缀）
    3. 将文件重命名为xxx_protein.pdb并移动到对应的子文件夹中
    """
    # 1. 获取所有xxx_p.pdb文件
    pdb_files = [f for f in os.listdir(source_dir)
                 if f.endswith("_p.pdb") and os.path.isfile(os.path.join(source_dir, f))]

    if not pdb_files:
        print(f"错误：在 {source_dir} 中未找到任何_p.pdb文件")
        return

    print(f"找到 {len(pdb_files)} 个需要整理的蛋白质文件：")
    for f in pdb_files:
        print(f"  - {f}")

    # 2. 逐个处理文件
    success_count = 0
    fail_count = 0
    for filename in pdb_files:
        try:
            # 提取核心文件夹名（去掉_p.pdb后缀，如5S8I_2LY_p.pdb → 5S8I_2LY）
            folder_name = filename.replace("_p.pdb", "")
            # 新文件名（将_p.pdb改为_protein.pdb，如5S8I_2LY_p.pdb → 5S8I_2LY_protein.pdb）
            new_filename = filename.replace("_p.pdb", "_protein.pdb")

            # 拼接路径
            old_file_path = os.path.join(source_dir, filename)  # 原文件路径
            folder_path = os.path.join(source_dir, folder_name)  # 子文件夹路径
            new_file_path = os.path.join(folder_path, new_filename)  # 新文件路径

            # 3. 创建子文件夹（如果不存在）
            if not os.path.exists(folder_path):
                if DRY_RUN:
                    print(f"[预览] 创建文件夹：{folder_path}")
                else:
                    os.makedirs(folder_path, exist_ok=True)
                    print(f"已创建文件夹：{folder_path}")
            else:
                print(f"文件夹已存在：{folder_path}")

            # 4. 检查目标文件是否已存在
            if os.path.exists(new_file_path):
                print(f"警告：目标文件已存在，跳过 → {new_file_path}")
                fail_count += 1
                continue

            # 5. 重命名并移动文件（预览/实际执行）
            if DRY_RUN:
                print(f"[预览] 重命名并移动：{old_file_path} → {new_file_path}")
            else:
                # 先重命名文件，再移动（或直接用shutil.move同时完成重命名+移动）
                shutil.move(old_file_path, new_file_path)
                print(f"已重命名并移动：{old_file_path} → {new_file_path}")
            success_count += 1

        except Exception as e:
            print(f"处理文件 {filename} 失败：{str(e)}")
            fail_count += 1

    # 6. 输出统计结果
    print("\n===== 整理完成 =====")
    print(f"总文件数：{len(pdb_files)}")
    print(f"成功整理：{success_count}")
    print(f"失败/跳过：{fail_count}")


# -------------------------- 执行主函数 --------------------------
if __name__ == "__main__":
    if DRY_RUN:
        print("=== 预览模式（不会实际修改文件）===\n")
    else:
        print("=== 实际执行模式（会创建文件夹并移动/重命名文件）===\n")

    reorganize_protein_files(SOURCE_DIR)
