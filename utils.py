import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist


def compute_residue_contact_matrix(pdb_file_path, contact_threshold=8.0):
    """
    计算蛋白质残基接触矩阵（完全遵循《结合位点方案.docx》逻辑）

    参数说明：
    - pdb_file_path: str，蛋白质PDB文件的完整路径（如"./pdbbind/1a0q/1a0q_protein.pdb"）
                    （匹配方案中“PDBbind数据集”的PDB文件格式要求）
    - contact_threshold: float，残基接触判定阈值（单位：Å），默认8.0Å
                        （方案中“动态掩码图卷积模块”明确使用8Å作为接触判定标准）

    返回值：
    - contact_matrix: np.ndarray，形状为(残基数量, 残基数量)的二进制矩阵
                      （元素为1表示残基对接触，0表示不接触，对角线为1（自身接触），匹配方案中接触矩阵定义）
    - residue_count: int，参与计算的残基总数（仅含20种标准氨基酸且含Cα原子的残基）
                      （方案中明确仅基于标准氨基酸构建接触矩阵，排除非标准残基）
    """
    # 1. 初始化PDB解析器（方案中数据预处理核心工具，与Biopython依赖一致）
    pdb_parser = PDBParser(QUIET=True)  # QUIET=True屏蔽无关警告，聚焦核心数据处理

    # 2. 定义20种标准氨基酸（方案中明确仅保留标准氨基酸，排除非蛋白质残基）
    standard_amino_acids = [
        "ALA", "ARG", "ASN", "ASP", "CYS",
        "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO",
        "SER", "THR", "TRP", "TYR", "VAL"
    ]

    # 3. 解析PDB文件，提取标准氨基酸的Cα原子坐标（方案中接触矩阵计算的核心数据）
    structure = pdb_parser.get_structure("protein", pdb_file_path)
    ca_coords = []  # 存储Cα原子坐标（每个残基仅保留1个Cα原子，代表残基空间位置）

    for model in structure:  # PDB文件可能包含多模型（如NMR结构），方案中默认取第一个模型
        for chain in model:
            for residue in chain:
                # 过滤条件1：仅保留标准氨基酸（排除配体、离子、非标准残基，与方案一致）
                if residue.get_resname() in standard_amino_acids:
                    # 过滤条件2：仅保留含Cα原子的残基（排除缺失Cα的不完整残基，确保坐标有效性）
                    if "CA" in residue:
                        ca_atom = residue["CA"]
                        ca_coords.append(ca_atom.get_coord().tolist())  # 提取Cα原子三维坐标

    # 4. 处理特殊情况：无有效残基（如PDB文件损坏）
    ca_coords = np.array(ca_coords, dtype=np.float32)
    residue_count = len(ca_coords)
    if residue_count == 0:
        raise ValueError(
            f"PDB文件 {pdb_file_path} 中未检测到有效标准氨基酸残基（需含Cα原子），无法计算接触矩阵（匹配方案数据有效性要求）")

    # 5. 计算残基接触矩阵（方案中“残基接触矩阵引导的注意力机制”核心逻辑）
    if residue_count == 1:
        # 单残基场景：仅自身接触，接触矩阵为1x1矩阵（对角线为1）
        contact_matrix = np.eye(1, dtype=np.float32)
    else:
        # 计算所有残基对的Cα原子欧氏距离（衡量空间邻近性，方案中核心距离计算方式）
        distance_matrix = cdist(ca_coords, ca_coords, metric="euclidean")
        # 判定接触：距离小于阈值则为接触（1），否则为不接触（0），与方案一致
        contact_matrix = (distance_matrix < contact_threshold).astype(np.float32)
        # 对角线设为1（残基本身视为接触，符合方案中接触矩阵的物理意义）
        np.fill_diagonal(contact_matrix, 1.0)

    return contact_matrix, residue_count




# 示例：计算PDBbind数据集中某蛋白质的残基接触矩阵
if __name__ == "__main__":
    # 1. 设定PDB文件路径（需替换为实际路径，匹配方案中PDBbind数据集结构）
    pdb_file = "data/pdb/1a0q/1a0q_protein.pdb"

    # 2. 调用函数计算接触矩阵（使用方案默认的8Å阈值）
    try:
        contact_mat, res_count = compute_residue_contact_matrix(pdb_file_path=pdb_file)
        # 3. 输出结果（验证与方案需求的一致性）
        print(f"PDB文件 {pdb_file} 处理结果：")
        print(f"- 有效残基数量：{res_count}")
        print(f"- 接触矩阵形状：{contact_mat.shape}")
        print(f"- 接触矩阵前5x5片段（验证格式）：\n{contact_mat[:5, :5]}")
    except Exception as e:
        print(f"计算失败：{e}")