import argparse
import glob
import os

def merge_common_files(file1_path, file2_path, output_dir):
    # 获取两个目录中的文件列表
    files_in_file1 = {f for f in os.listdir(file1_path) if f.endswith('.txt')}
    files_in_file2 = {f for f in os.listdir(file2_path) if f.endswith('.txt')}

    # 找到共有的文件
    common_files = files_in_file1.intersection(files_in_file2)

    for filename in common_files:
        # 构建文件的完整路径
        file1_full_path = os.path.join(file1_path, filename)
        file2_full_path = os.path.join(file2_path, filename)

        # 输出文件的路径和名称
        output_full_path = os.path.join(output_dir, filename)

        try:
            # 读取文件内容
            with open(file1_full_path, 'r', encoding='utf-8') as f1:
                content1 = f1.readlines()

            with open(file2_full_path, 'r', encoding='utf-8') as f2:
                content2 = f2.readlines()

            # 合并内容（这里简单拼接，可以根据需要调整合并逻辑）
            merged_content = content1 + content2

            # 写入输出文件
            with open(output_full_path, 'w', encoding='utf-8') as f_out:
                f_out.writelines(merged_content)

            print(f"Merged {filename} successfully.")

        except FileNotFoundError:
            print(f"Error: {filename} not found in one of the directories.")
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', default='ranked_rules')
    parser.add_argument('--file2', default='ranked_rules2')
    parser.add_argument('--output_folder', default='merge_rules')
    parser.add_argument("--dataset", default="family")
    parser.add_argument("-p", default='gpt-3.5-turbo-top-0-f-50-l-10', help="rule path")
    args = parser.parse_args()

    file1_path = os.path.join(args.file1, args.dataset, args.p)
    file2_path = os.path.join(args.file2, args.dataset)
    output_folder = os.path.join(args.output_folder, args.dataset, args.p)
    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    merge_common_files(file1_path, file2_path, output_folder)
