import os
import argparse
import glob
from tqdm import tqdm
from data import *
import json


def parse_rule(r):
    """parse a rule into body and head"""
    head, body = r.split(" <-- ")
    head_list = head.split("\t")
    score = [float(s) for s in head_list[:-1]]
    head = head_list[-1]
    body = body.split(", ")
    return score, head, body


def load_rules(rule_path):
    all_rules = {}

    for input_filepath in glob.glob(os.path.join(rule_path, "*_sorted_rules.txt")):
        with open(input_filepath, 'r') as f:
            rules = f.readlines()
            for i_, rule in enumerate(rules):

                score, head, body = parse_rule(rule.strip('\n'))

                if head not in all_rules:
                    all_rules[head] = []
                all_rules[head].append(body)
    return all_rules


def evaluate_rule(rule_body, rule_head, fact_dict, r2mat, e_num, ent2idx):
    score = {}
    r_size = len(fact_dict[rule_head])
    support = 0
    pca_negative = 0

    # Rule reachable matrix
    path_count = sparse.eye(e_num)
    for b_rel in rule_body:
        # print(f"path_count shape: {path_count.shape}, r2mat[b_rel] shape: {r2mat[b_rel].shape}")
        path_count = path_count * r2mat[b_rel]

    visted_head = set()
    for fact in fact_dict[rule_head]:
        h, _, t = parse_rdf(fact)
        if path_count[ent2idx[h], ent2idx[t]] != 0:
            support += 1
        visted_head.add(h)

    if support == 0:
        return {"support": 0., "coverage": 0., "confidence": 0., "pca_confidence": 0.}

    for head in visted_head:
        pca_negative += path_count[ent2idx[head], :].count_nonzero()

    all_path = path_count.count_nonzero()

    score['support'] = support
    score['coverage'] = support / r_size
    score['confidence'] = support / all_path
    score['pca_confidence'] = support / pca_negative
    return score


def main(args):
    input_folder = os.path.join(args.input_path, args.dataset, args.p)
    # print(input_folder)
    rule = load_rules(input_folder)
    # print('rule:',rule)
    # print('rule长度:',len(rule))
    dataset = Dataset(data_root='datasets/{}/'.format(args.dataset), inv=True)
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    # print('all_rdf的内容:',all_rdf)
    test_rdf = all_rdf if args.eval_mode == "all" else dataset.test_rdf
    fact_dict = construct_fact_dict(test_rdf)
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
    # print(f"r2mat dimensions: {[m.shape for m in r2mat.values()]}")

    output_folder = os.path.join(args.output_path, args.dataset, args.p)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_statics = {"support": 0., "coverage": 0., "confidence": 0., "pca_confidence": 0.}

    for r_head in tqdm(rule):
        if r_head not in fact_dict:
            if args.eval_mode == "all":
                raise ValueError("Rule head {} not in fact set. Please have a check.".format(r_head))
            else:
                continue
        if args.debug:
            print("Rule head: {}".format(r_head))
        rule_statics = {"support": 0., "coverage": 0., "confidence": 0., "pca_confidence": 0.}
        file_name = r_head.replace('/', '-')
        with open(os.path.join(output_folder, "{}.txt".format(file_name)), 'w') as f:
            for rule_body in rule[r_head]:
                score = evaluate_rule(rule_body, r_head, fact_dict, r2mat, e_num, ent2idx)
                if args.debug:
                    print(f"Rule body: {rule_body}, score: {score}")
                f.write(
                    f"{score['support']}\t{score['coverage']}\t{score['confidence']}\t{score['pca_confidence']}\t{r_head} <-- {', '.join(rule_body)}\n")

                # Add statiscs
                for k in score:
                    rule_statics[k] += score[k]

        with open(os.path.join(output_folder, "{}_rule_statics.json".format(file_name)), 'w') as f:
            for k in rule_statics:
                rule_statics[k] /= len(rule[r_head])  # 计算出每个规则头统计指标的平均值
                data_statics[k] += rule_statics[k]
            json.dump(rule_statics, f, indent=2)
        if args.debug:
            print("Rule {} statics: {}".format(r_head, rule_statics))
    with open(os.path.join(output_folder, "data_statics.json"), 'w') as f:
        for k in data_statics:
            data_statics[k] /= len(rule)  # 得到数据集级别的平均统计指标
        json.dump(data_statics, f, indent=2)
    # print("Data statics: {}".format(data_statics))
    print("support	coverage	confidence	pca_confidence")
    print(
        f"{data_statics['support']}	{data_statics['coverage']}	{data_statics['confidence']}	{data_statics['pca_confidence']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="family")
    parser.add_argument("-p", default="gpt-3.5-turbo-top-0-f-50-l-10")
    # parser.add_argument("-t", default="0121", help="The date on which the code was run")
    parser.add_argument("--eval_mode", choices=['all', "test", 'fact'], default="all",
                        help="evaluate on all or only test set")
    parser.add_argument("--input_path", default="sorted_rules", type=str, help="input folder")
    parser.add_argument("--output_path", default="ranked_rules_last", type=str, help="path to output file")
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    main(args)
