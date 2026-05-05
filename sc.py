import csv


def read_csv_sequences(csv_file):
    """读取CSV中的所有序列，返回集合用于快速匹配"""
    sequences = set()
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'Sequence' not in reader.fieldnames:
                raise ValueError("CSV文件中未找到'Sequence'列，请检查列名")

            for row in reader:
                seq = row['Sequence'].strip()
                if seq:
                    sequences.add(seq)
        print(f"从CSV中读取到 {len(sequences)} 个有效序列")
        return sequences
    except Exception as e:
        print(f"读取CSV失败：{e}")
        return set()


def process_train_file(train_file, csv_sequences, output_train1, output_test):
    """处理train.txt，按“两行一组”分割，输出无空行的结果"""
    try:
        # 读取所有行，过滤空行（原始文件中的空行不影响分组）
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # 确保行数为偶数（两行一组）
        if len(lines) % 2 != 0:
            print(f"警告：train.txt总行数为 {len(lines)}（奇数），最后一行将被忽略")
            lines = lines[:-1]

        # 按“两行一组”分组（序列行 + 01串行）
        data_pairs = []
        for i in range(0, len(lines), 2):
            seq_line = lines[i]  # 序列行
            binary_line = lines[i + 1]  # 01串行
            data_pairs.append((seq_line, binary_line))

        print(f"从train.txt中解析到 {len(data_pairs)} 组数据（两行一组）")

        # 分离匹配和不匹配的组
        train1_pairs = []  # 不匹配的组
        test_pairs = []  # 匹配的组
        matched_count = 0

        for seq, binary in data_pairs:
            if seq in csv_sequences:
                test_pairs.append((seq, binary))
                matched_count += 1
            else:
                train1_pairs.append((seq, binary))

        print(f"匹配到 {matched_count} 组数据（序列存在于CSV中）")

        # 写入结果文件（无空行：组内两行连续，组间也无空行）
        with open(output_train1, 'w', encoding='utf-8') as f:
            for seq, binary in train1_pairs:
                f.write(seq + '\n')  # 序列行
                f.write(binary + '\n')  # 01串行（写完直接接下一组，无空行）

        with open(output_test, 'w', encoding='utf-8') as f:
            for seq, binary in test_pairs:
                f.write(seq + '\n')
                f.write(binary + '\n')

        print(
            f"处理完成：\n- 不匹配的 {len(train1_pairs)} 组数据已保存到 {output_train1}\n- 匹配的 {len(test_pairs)} 组数据已保存到 {output_test}")
        print("注意：输出文件中所有行连续排列，无空行")

    except Exception as e:
        print(f"处理train.txt失败：{e}")


if __name__ == '__main__':
    # 配置文件路径
    csv_path = r'E:\project\BindSite\data\test.csv'  # CSV路径
    train_path = 'train.txt'  # train.txt路径（绝对路径如 r'C:\xxx\train.txt'）
    train1_path = 'train1.txt'
    test_path = 'test1.txt'

    csv_seqs = read_csv_sequences(csv_path)
    if csv_seqs:
        process_train_file(train_path, csv_seqs, train1_path, test_path)