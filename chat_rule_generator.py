import argparse
import json

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


def build_prompt(head, candidate_rels, is_zero, k):
    # head = clean_symbol_in_rel(head)
    instruction = (
        "Logical rules define the relationship between two entities: X and Y. Each rule is written in the form "
        "of a logical implication, which states that if the conditions on the right-hand side (rule body) are "
        "satisfied, then the statement on the left-hand side (rule head) holds true.\n"
    )

    if is_zero and args.k != 0:  # Zero-shot
        context = """For examples:
        husband(X,Y) <-- father(X, Z_1) & inv_mother(Z_1, Y) // X is the husband of Y, if X is the father of Z_1, and  Y is the mother of Z_1
        husband(X,Y) <-- father(X, Z_1) & son(Z_1, Y) // X is the husband of Y, if X is the father of Z_1, and Z_1 is the son of Y.
        husband(X,Y) <-- father(X, Z_1) & sister(Z_1, Z_2) & daughter(Z_2, Y) // X is the husband of Y, if X is the father of Z_1, Z_1 is the brother of Z_2, and Z_2 is the daughter of Y.
        """
        predict = f'\nGiven a rule head: "{head}(X,Y)", please generate {k} rules that are the most important and relevant to the rule head.'
    else:  # Few-shot
        context = "Samples:\n"
        background = "\nBackground:\n"
        if args.k != 0:
            predict = f'\nBased on the above rules and background, please generate {k} rules that are most important to the rule head: "{head}(X,Y)". Return the rules only without any explanations.'
        else:
            predict = f'\nBased on the above rules and background, please generate as many of the most important rules for the rule head: "{head}(X,Y)" as possible. Return the rules only without any explanations.'
    predict += ("\nPlease only select predicates form: {}. Return the ru"
                "les only without any explanations.").format(
        candidate_rels
    )
    return instruction, context, background, predict

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

def generate_rule(candidate_rels, rule_path, model, args, sampled_path):
    head = sampled_path['head']
    paths = sampled_path["paths"]

    # print("Head: ", head)

    # Raise an error if k=0 for zero-shot setting
    if args.k == 0 and args.is_zero:
        raise NotImplementedError(
            f"""Cannot implement for zero-shot(f=0) and generate zero(k=0) rules."""
        )
    # Build prompt excluding rules
    instruction, context, background, predict = build_prompt(
        head, candidate_rels, args.is_zero, args.k
    )
    current_prompt = instruction + context + background + predict

    if args.is_zero:  # For zero-shot setting
        with open(os.path.join(rule_path, f"{head}_zero_shot.query"), "w") as f:
            f.write(current_prompt + "\n")
            f.close()
        if not args.dry_run:
            response = query(current_prompt, model=args.model_name)
            with open(os.path.join(rule_path, f"{head}_zero_shot.txt"), "w") as f:
                f.write(response + "\n")
                f.close()
    else:  # For few-shot setting
        path_content_list = modify_path_format(paths, head)
        # print('path_content_list: ', path_content_list)
        file_name = head.replace("/", "-")
        with open(os.path.join(rule_path, f"{file_name}.txt"), "w") as rule_file, open(
            os.path.join(rule_path, f"{file_name}.query"), "w"
        ) as query_file:
            rule_file.write(f"Rule_head: {head}\n")
            for i in range(args.l):
                relation_meaning_path = f'relation_meaning/{args.dataset}/{args.model_name}/combined.txt'
                with open(relation_meaning_path, 'r', encoding='utf-8') as file:
                    relation_meaning = file.read()
                few_shot_samples = random.sample(
                    path_content_list, min(args.f, len(path_content_list))
                )
                few_shot_paths = check_prompt_length(
                    current_prompt, few_shot_samples, relation_meaning, model
                )
                prompt = instruction + context + few_shot_paths + background + relation_meaning + predict  # Prompt
                # tqdm.write("Prompt: \n{}".format(prompt))
                query_file.write(f"Sample {i + 1} time: \n")
                query_file.write(prompt + "\n")
                if not args.dry_run:
                    response = model.generate_sentence(prompt)
                    # tqdm.write("Response: \n{}".format(response))
                    rule_file.write(f"Sample {i + 1} time: \n")
                    rule_file.write(response + "\n")

def main(args, LLM):
    data_path = os.path.join(args.data_path, args.dataset) + "/"
    dataset = Dataset(data_root=data_path, inv=True)
    sampled_path_dir = os.path.join(args.data_path, args.dataset)
    sampled_path = read_paths(os.path.join(sampled_path_dir, "closed_rel_paths.jsonl"))
    # print(sampled_path_dir,sampled_path)
    rdict = dataset.get_relation_dict()
    all_rels = list(rdict.rel2idx.keys())
    # print('all_rels: ',all_rels)
    candidate_rels = ", ".join(all_rels)
    # print('candidate_rels: ',candidate_rels)
    # Save paths
    rule_path = os.path.join(
        args.rule_path,
        args.dataset,
        f"{args.prefix}{args.model_name}-top-{args.k}-f-{args.f}-l-{args.l}",
    )
    if not os.path.exists(rule_path):
        os.makedirs(rule_path)

    model = LLM(args)
    print("Prepare pipline for inference...")
    model.prepare_for_inference()

    # Generate rules
    for path in tqdm(sampled_path, total=len(sampled_path)):
        generate_rule(candidate_rels, rule_path, model, args, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="datasets", help="data directory"
    )
    parser.add_argument("--dataset", type=str, default="family", help="dataset")
    parser.add_argument(
        "--rule_path", type=str, default="gen_rules", help="path to rule file"
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
    parser.add_argument('-p', default='gpt-3.5-turbo-l-10', help='rule prefix')
    parser.add_argument("-f", type=int, default=5, help="Few-shot number")
    parser.add_argument(
        "-l", type=int, default=3, help="sample l times for generating k rules"
    )
    parser.add_argument("--prefix", type=str, default="", help="prefix")
    parser.add_argument("--dry_run", action="store_true", help="dry run")

    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    main(args, LLM)
