import argparse
import os

from data import *
from llms import get_registed_model
from tqdm import tqdm

def generate_paths(head, model, output_path):
    # print(head)
    # Build prompt
    predict = f'In a knowledge graph, Please provide an explanation of the meaning of the relation {head} with one sentence:'
    # predict = f'In a biomedical knowledge graph, Please provide an explanation of the meaning of the relation {head} with one sentence:'
    # predict = f'In a English vocabulary knowledge graph, Please provide an explanation of the meaning of the relation {head} with one sentence:'

    with open(os.path.join(output_path, f"{head}.txt"), "w") as relation_meaning:
        response = model.generate_sentence(predict)
        relation_meaning.write(response + "\n")

def main(args,LLM):
    data_path = os.path.join(args.data_path, args.dataset) + '/'
    dataset = Dataset(data_root=data_path, inv=True)
    rdict = dataset.get_relation_dict()
    # print('rdict长度:',rdict.__len__())
    all_rels = list(rdict.rel2idx.keys())
    # Save paths
    output_path = os.path.join(
        args.output_path,
        args.dataset,
        f"{args.prefix}{args.model_name}",
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model = LLM(args)
    print("Prepare pipline for inference...")
    model.prepare_for_inference()
    for head in tqdm(rdict.rel2idx):
        if head == "None" or "inv_" in head:
            continue

        generate_paths(head, model, output_path)

    # 定义目标文件路径
    output_file = os.path.join(output_path, "combined.txt")

    # 打开目标文件，准备追加内容
    with open(output_file, 'a', encoding='utf-8') as outfile:
        # 遍历目录中的所有文件
        for filename in os.listdir(output_path):
            # 检查文件是否以 .txt 结尾
            if filename.endswith('.txt'):
                # 构建文件的完整路径
                file_path = os.path.join(output_path, filename)
                # 打开当前文件并读取内容
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # 将当前文件的内容追加到目标文件中
                    outfile.write(infile.read())
                    # 可选：在每个文件内容之间添加一个换行符
                    outfile.write('\n')

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='data directory')
    parser.add_argument('--dataset', type=str, default='family', help='dataset')
    parser.add_argument("--output_path", type=str, default="relation_meaning", help="output path")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="model name")
    parser.add_argument("--prefix", type=str, default="", help="prefix")

    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    main(args, LLM)