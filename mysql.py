import pymysql
import csv
import os

# ===================== 【仅修改这里】你的配置 =====================
MYSQL_HOST = "218.194.61.110"       # 主机(默认不变)
MYSQL_PORT = 3306              # 端口(默认不变)
MYSQL_USER = "root"            # 用户名(默认不变)
MYSQL_PASSWORD = "k$RSwf#gnvku5S#hdvAS" # 改成你的MySQL密码！
MYSQL_DATABASE = "kdecm3"      # 固定：你的库名
CSV_PATH = "./data/train.csv"       # 改成你的train.csv真实路径！
TABLE_NAME = "zhongyao_huayao_pdb_bindsite" # 固定：表名
# ==================================================================

def import_data_to_mysql():
    # 校验文件
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到文件：{CSV_PATH}")
        return

    # 连接MySQL
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        print("✅ 连接 MySQL 成功！")
    except Exception as e:
        print(f"❌ 连接失败：{e}")
        return

    success = 0
    fail = 0

    # 读取CSV并导入
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pdb_id = row['PDB']
                sequence = row['Sequence']
                bs = row['BS']

                # 插入SQL（重复ID自动更新）
                sql = f"""
                INSERT INTO {TABLE_NAME} (pdb_id, sequence, binding_sites)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                sequence=%s, binding_sites=%s
                """
                cursor.execute(sql, (pdb_id, sequence, bs, sequence, bs))
                success += 1
            except Exception as e:
                fail += 1
                print(f"⚠️ 导入失败 {row['PDB']}：{e}")

    # 提交并关闭
    conn.commit()
    cursor.close()
    conn.close()

    # 结果打印
    print("="*60)
    print(f"✅ 导入完成！")
    print(f"📊 数据库：kdecm3")
    print(f"📊 表名：{TABLE_NAME}")
    print(f"✅ 成功：{success} 条")
    print(f"❌ 失败：{fail} 条")
    print("="*60)

if __name__ == "__main__":
    import_data_to_mysql()