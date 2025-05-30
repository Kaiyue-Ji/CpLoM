import json
import os.path
import pickle
from data import *
import re
import torch.multiprocessing as mp
import numpy as np
from scipy import sparse
from collections import defaultdict
import argparse
from utils import *
import glob
from tqdm import tqdm

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

head2mrr = defaultdict(list)
head2hit_5 = defaultdict(list)
head2hit_10 = defaultdict(list)
head2hit_1 = defaultdict(list)


def sortSparseMatrix(m, r, rev=True, only_indices=False):
    """ Sort a row in matrix row and return column index
    """
    d = m.getrow(r)
    s = zip(d.indices, d.data)
    sorted_s = sorted(s, key=lambda v: v[1], reverse=rev)
    if only_indices:
        res = [element[0] for element in sorted_s]
    else:
        res = sorted_s
    return res


def remove_var(r):
    """R1(A, B), R2(B, C) --> R1, R2"""
    r = re.sub(r"\(\D?, \D?\)", "", r)
    return r


def parse_rule(r):
    """parse a rule into body and head"""
    head, body = r.split(" <-- ")
    head_list = head.split("\t")
    score = [float(s) for s in head_list[:-1]]
    head = head_list[-1]
    body = body.split(", ")
    return score, head, body


def load_rules(rule_path, all_rules, all_heads):
    for input_filepath in glob.glob(os.path.join(rule_path, "*.txt")):
        with open(input_filepath, 'r') as f:
            rules = f.readlines()
            for i_, rule in enumerate(rules):
                score, head, body = parse_rule(rule.strip('\n'))
                # Skip zero support rules
                if score[0] == 0.0:
                    continue
                if head not in all_rules:
                    all_rules[head] = []
                all_rules[head].append((head, body, score))

                if head not in all_heads:
                    all_heads.append(head)


def get_gt(dataset):
    # entity
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    # global fact_rdf, train_rdf, valid_rdf, test_rdf# = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    gt = defaultdict(list)
    all_rdf = fact_rdf + train_rdf + valid_rdf + test_rdf
    for rdf in all_rdf:
        h, r, t = parse_rdf(rdf)
        if h not in ent2idx:
            ent2idx[h] = len(ent2idx)
            idx2ent[len(idx2ent)] = h
        if t not in ent2idx:
            ent2idx[t] = len(ent2idx)
            idx2ent[len(idx2ent)] = t
        assert h in ent2idx, f"Entity {h} not found in ent2idx."
        assert t in ent2idx, f"Entity {t} not found in ent2idx."
        gt[(h, r)].append(ent2idx[t])
    return gt

# 定义一个函数用于保存进度
def save_progress(progress_file, progress_data):
    """
    保存进度到文件
    :param progress_file: 进度日志文件路径
    :param progress_data: 包含所有 head 进度的字典
    """
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=4)  # 使用 indent 参数使日志文件更易读

def kg_completion(rules, dataset, args):
    """
    Input a set of rules
    Complete Querys from test_rdf based on rules and fact_rdf
    """
    # # rdf_data
    # global fact_rdf, train_rdf, valid_rdf, test_rdf
    # fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    # global all_rdf
    all_rdf = fact_rdf + train_rdf + valid_rdf
    # # groud truth 真值
    gt = get_gt(dataset)
    # relation
    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()
    rel2idx, idx2rel = rdict.rel2idx, rdict.idx2rel
    # entity
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx

    for rdf in all_rdf:
        h, _, t = parse_rdf(rdf)
        if h not in ent2idx:
            ent2idx[h] = len(ent2idx)
            idx2ent[len(idx2ent)] = h
        if t not in ent2idx:
            ent2idx[t] = len(ent2idx)
            idx2ent[len(idx2ent)] = t

    e_num = len(idx2ent)
    for rdf in all_rdf:
        h, _, t = parse_rdf(rdf)
        assert h in ent2idx, f"Entity {h} not found in ent2idx."
        assert t in ent2idx, f"Entity {t} not found in ent2idx."

    # construct relation matrix (following Neural-LP)
    r2mat = construct_rmat(idx2rel, idx2ent, ent2idx, all_rdf)
    # Test rdf grouped by head
    test = {}
    for rdf in test_rdf:
        query = parse_rdf(rdf)
        q_h, q_r, q_t = query
        if q_r not in test:
            test[q_r] = [query]
        else:
            test[q_r].append(query)

    # total_queries = len(test['isAffiliatedTo'])
    # print(f"Total queries for head 'isAffiliatedTo': {total_queries}")
    # output_folder = os.path.join(args.output_path, args.dataset, args.p)
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # output_pred_path =os.path.join(output_folder, 'output_predict.pkl')

    mrr, hits_1, hits_5, hits_10 = [], [], [], []

    # if args.rank_only and os.path.exists(output_pred_path):
    #     with open(output_pred_path, 'rb') as f:
    #         output_pred = pickle.load(f)
    # else:
    output_pred = {}

    score_name_to_id = {"support": 0, "coverage": 1, "confidence": 2, "pca_confidence": 3, "none": -1}
    score_id = score_name_to_id[args.score]
    threshold_score_id = score_name_to_id[args.threshold_score]

    # print('总任务数:',len(test.keys()))
    for head in tqdm(test.keys(), desc="Processing heads"):
        # if head != 'isAffiliatedTo':
        #     continue

        # 检查是否需要跳过（例如没有规则）
        if head not in rules:
            # print(f"Skipping head '{head}' because it has no rules.")
            continue

        # 检查查询列表是否为空
        if len(test[head]) == 0:
            # print(f"Head '{head}' has no queries. Marking as completed.")
            # heads_progress[head] = {"last_query_index": -1}
            # processed_heads_count += 1
            continue
        if not args.rank_only:
            output_pred[head] = {}
        if head not in rules:
            continue
        _rules = rules[head]
        # print('打印:',_rules)
        # print('输出args.rank_only:',args.rank_only)
        if not args.rank_only:
            # with open(os.path.join(input_folder, f"{head}_sorted_rules.jsonl"), "w") as sorted_path:
            path_count = sparse.dok_matrix((e_num, e_num))
            if score_id != -1:
                sorted_rules = sorted(_rules, key=lambda x: x[2][score_id], reverse=True)
                # json.dump(sorted_rules, sorted_path)
                # sorted_path.write("\n")
                if args.top > 0:
                    _rules = sorted_rules[:args.top]
                if args.threshold > 0:
                    _rules = [rule for rule in sorted_rules if rule[2][threshold_score_id] > args.threshold]
            for rule in _rules:
                head, body, score = rule
                if score_id != -1:
                    score = score[score_id]
                else:
                    score = 1.0
                body_adj = sparse.eye(e_num)
                for b_rel in body:
                    body_adj = body_adj * r2mat[b_rel]

                body_adj = body_adj * score
                # print(f"body_adj shape: {body_adj.shape}")
                # print('赋予规则的权重:',body_adj)
                path_count += body_adj
                # if args.debug:
                #     print("path_count: ", path_count)


        # 查询处理部分
        # for q_i, query_rdf in enumerate(
        #         tqdm(test[head][start_index:], desc=f"Processing queries for head '{head}'", leave=False),
        #         start=start_index):
        for q_i, query_rdf in enumerate(test[head]):
            # print("query rdf: ", query_rdf)
            query = parse_rdf(query_rdf)
            q_h, q_r, q_t = query

            if args.debug:
                print("{}\t{}\t{}".format(q_h, q_r, q_t))

            if not args.rank_only:
                pred = np.squeeze(np.array(path_count[ent2idx[q_h]].todense()))
                output_pred[head][(q_h, q_r, q_t)] = pred
            else:
                pred = output_pred[head][(q_h, q_r, q_t)]
            #
            # # print('查看pred相关内容:',pred)
            # # ill-rank
            if args.rank_mode == 'ill':
                rank = ill_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            elif args.rank_mode == 'harsh':
                rank = harsh_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            elif args.rank_mode == 'balance':
                rank = balance_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            else:
                rank = random_rank(pred, gt, ent2idx, q_h, q_t, q_r)
            # rank = 140000
            mrr.append(1.0 / rank)
            head2mrr[q_r].append(1.0 / rank)

            hits_1.append(1 if rank <= 1 else 0)
            hits_5.append(1 if rank <= 5 else 0)
            hits_10.append(1 if rank <= 10 else 0)
            head2hit_1[q_r].append(1 if rank <= 1 else 0)
            head2hit_5[q_r].append(1 if rank <= 5 else 0)
            head2hit_10[q_r].append(1 if rank <= 10 else 0)
            if args.debug:
                print("rank at {}: {}".format(q_i, rank))

    # if not args.rank_only:
    #     with open(output_pred_path, 'wb') as f:
    #         pickle.dump(output_pred, f)

    return mrr, hits_1, hits_5, hits_10


def load_results(head):
    input_file_name = os.path.join(args.output_path, args.dataset, args.p, 'output_predict.pkl')
    with open(input_file_name, 'rb') as f:
        pred_results_dict = pickle.load(f)
    return pred_results_dict[head]


def feq(relation, fact_rdf):
    count = 0
    for rdf in fact_rdf:
        h, r, t = parse_rdf(rdf)
        if r == relation:
            count += 1
    return count




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='ranked_rules')
    # parser.add_argument('--sorted_rules', default='sorted_rules')
    parser.add_argument("--dataset", default="family")
    parser.add_argument('--output_path', default='pred_results1', type=str, help='path to save pred results')
    parser.add_argument("-p", default='gpt-3.5-turbo-top-0-f-50-l-10', help="rule path")
    parser.add_argument("--eval_mode", choices=['all', "test", 'fact'], default="all",
                        help="evaluate on all or only test set")
    parser.add_argument('--cpu_num', type=int, default=mp.cpu_count() // 2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rank_mode", choices=['ill', 'harsh', 'balance'], default='harsh')
    parser.add_argument("--rank_only", action="store_true")
    parser.add_argument("--threshold_score", choices=['pca_confidence', 'confidence', 'coverage', 'support'],
                        default='support')
    parser.add_argument("--score", choices=['pca_confidence', 'confidence', 'coverage', 'support', 'none'],
                        default='pca_confidence')
    args = parser.parse_args()
    dataset = Dataset(data_root='datasets/{}/'.format(args.dataset), inv=True)

    all_rules = {}
    all_rule_heads = []

    # sorted_rules = os.path.join(args.sorted_rules, args.dataset, args.p, f"rank_top-{args.top}")
    # if not os.path.exists(sorted_rules):
    #     os.makedirs(sorted_rules)

    input_folder = os.path.join(args.input_folder, args.dataset, args.p)
    # input_folder = os.path.join(args.input_folder, args.dataset)

    print("Rule path is {}".format(input_folder))
    load_rules(input_folder, all_rules, all_rule_heads)

    # fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    # rdf_data
    global fact_rdf, train_rdf, valid_rdf, test_rdf
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf

    test_mrr, test_hits_1, test_hits_5, test_hits_10 = kg_completion(all_rules, dataset, args)

    if args.debug:
        print_msg("distribution of test query")
        for head in all_rule_heads:
            count = feq(head, test_rdf)
            print("Head: {} Count: {}".format(head, count))

        print_msg("distribution of train query")
        for head in all_rule_heads:
            count = feq(head, fact_rdf + valid_rdf + train_rdf)
            print("Head: {} Count: {}".format(head, count))

        all_results = {"mrr": [], "hits_1": [], "hits_5": [], "hits_10": []}
        print_msg("Stat on head and hit@1")
        for head, hits in head2hit_1.items():
            print(head, np.mean(hits))
            all_results["hits_1"].append(np.mean(hits))

        print_msg("Stat on head and hit@5")
        for head, hits in head2hit_5.items():
            print(head, np.mean(hits))
            all_results["hits_5"].append(np.mean(hits))

        print_msg("Stat on head and hit@10")
        for head, hits in head2hit_10.items():
            print(head, np.mean(hits))
            all_results["hits_10"].append(np.mean(hits))

        print_msg("Stat on head and mrr")
        for head, mrr in head2mrr.items():
            print(head, np.mean(mrr))
            all_results["mrr"].append(np.mean(mrr))
    dataset_name = args.dataset + ": " + args.p

    result_dict = {"mrr": np.mean(test_mrr), "hits_1": np.mean(test_hits_1), "hits_5": np.mean(test_hits_5),
                   "hits_10": np.mean(test_hits_10)}
    output_dir = os.path.join(args.output_path, args.dataset, args.p, f'arg.top-{str(args.top)}')
    # print('输出目录output_dir:',output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "result_dict.json"), 'w') as f:
        json.dump(result_dict, f)
    # print("{}: MRR: {} Hits@1: {} Hits@5: {} Hits@10: {}".format(dataset_name, np.mean(test_mrr), np.mean(test_hits_1), np.mean(test_hits_5),np.mean(test_hits_10)))
    print("MRR	Hits@1	Hits@5	Hits@10")
    print("{}	{}	{}	{}".format(np.mean(test_mrr), np.mean(test_hits_1), np.mean(test_hits_5),
                                        np.mean(test_hits_10)))