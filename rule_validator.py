import argparse
import json
import re

from tqdm import tqdm
from data import *

from utils import *
from llms import get_registed_model


def read_paths(path):
    results = []
    # print(path)
    with open(path, "r") as f:
        # print(f.readlines())
        # for line in f.readlines():
        for line in f:
            # print('line;',line)
            results.append(json.loads(line.strip()))
    return results


def build_prompt(is_zero, k):
    # head = clean_symbol_in_rel(head)

    if is_zero and args.k != 0:  # Zero-shot
        context = """For examples:
        husband(X,Y) <-- father(X, Z_1) & inv_mother(Z_1, Y)
        husband(X,Y) <-- father(X, Z_1) & son(Z_1, Y)
        husband(X,Y) <-- father(X, Z_1) & sister(Z_1, Z_2) & daughter(Z_2, Y)
        """
    else:  # Few-shot
        context = "Rule:\n"
    instruction = (
        "where inv_relation(X,Y) means that relation(Y, X) holds."
        "relation(Y, X) means that Y has a relationship with X."
        "relation(X, Y) means that X has a relationship with Y."
        "please give the answer: correct or wrong\n\n"
    )
    tips = ("attention: do not need give the reason.\n"
            )
    return context, instruction, tips

def modify_path_format(path, head):
    """
    Modify path format for prompt, return a list of path in new format
    """
    path_list = []
    # head = clean_symbol_in_rel(head)
    for p in path:
        context = f"{head}(X,Y) <-- "
        for i, r in enumerate(p.split("|")):
            # r = clean_symbol_in_rel(r)
            if i == 0:
                first = "X"
            else:
                first = f"Z_{i}"
            if i == len(p.split("|")) - 1:
                last = "Y"
            else:
                last = f"Z_{i + 1}"
            context += f"{r}({first}, {last}) & "
        context = context.strip(" & ")
        path_list.append(context)
    return path_list

def validate_rule(rule_path, model, args, sampled_path):
    # print(sampled_path)
    head = sampled_path['head']
    paths = sampled_path['paths']

    valid_rules = []
    # Build prompt excluding rules
    context, instruction, tips = build_prompt(args.is_zero, args.k)
    first_prompt = context + instruction + tips
    # Raise an error if k=0 for zero-shot setting
    if args.k == 0 and args.is_zero:
        raise NotImplementedError(
            f"""Cannot implement for zero-shot(f=0) and generate zero(k=0) rules."""
        )
    if args.is_zero:  # For zero-shot setting
        with open(os.path.join(rule_path, f"{head}_zero_shot.query"), "w") as f:
            f.write(first_prompt + "\n")
            f.close()
        if not args.dry_run:
            response = query(first_prompt, model=args.model_name)
            with open(os.path.join(rule_path, f"{head}_zero_shot.txt"), "w") as f:
                f.write(response + "\n")
                f.close()
    else:  # For few-shot setting
        path_content_list = modify_path_format(paths, head)
        file_name = head.replace("/", "-")
        with open(os.path.join(rule_path, f"{file_name}.query"), "w") as query_file, open(
                os.path.join(rule_path, f"{file_name}.txt"), "w", encoding='utf-8') as rule_file:
            for rule in path_content_list:
                prompt = context + rule + "\n" + instruction + tips
                query_file.write(prompt + "\n")
                rule_file.write(f"Rule: {rule}\n")

                if not args.dry_run:
                    try:
                        response = model.generate_sentence(prompt)
                        rule_file.write("answer: " + response + "\n")
                    except Exception as e:
                        print(f"An error occurred while writing to the file: {e}")



def main(args, LLM):
    data_path = os.path.join(args.data_path, args.dataset) + "/"
    dataset = Dataset(data_root=data_path, inv=True)
    sampled_path_dir = os.path.join(args.data_path, args.dataset)
    sampled_path = read_paths(os.path.join(sampled_path_dir, "closed_rel_paths.jsonl"))
    # print(sampled_path_dir,sampled_path)
    rdict = dataset.get_relation_dict()
    all_rels = list(rdict.rel2idx.keys())
    # print('all_rels: ',all_rels)
    # Save paths
    rule_path = os.path.join(
        args.rule_path,
        args.dataset,
        f"{args.prefix}{args.model_name}-top-{args.k}",
    )
    if not os.path.exists(rule_path):
        os.makedirs(rule_path)

    model = LLM(args)
    print("Prepare pipline for inference...")
    model.prepare_for_inference()

    # Generate rules
    for path in tqdm(sampled_path, total=len(sampled_path)):
        validate_rule(rule_path, model, args, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="datasets", help="data directory"
    )
    parser.add_argument("--dataset", type=str, default="family", help="dataset")
    parser.add_argument(
        "--rule_path", type=str, default="filter_rules", help="path to rule file"
    )
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="model name")
    parser.add_argument(
        "--is_zero",
        action="store_true",
        help="Enable this for zero-shot rule generation",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=0,
        help="Number of generated rules, 0 denotes as much as possible",
    )
    parser.add_argument("--prefix", type=str, default="", help="prefix")
    parser.add_argument("--dry_run", action="store_true", help="dry run")

    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    main(args, LLM)
