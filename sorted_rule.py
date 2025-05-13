
import argparse
import glob
import os

import glob
import os


def parse_rule(r):
    """Parse a rule into body and head"""
    head, body = r.split(" <-- ")
    head_list = head.split("\t")
    score = [float(s) for s in head_list[:-1]]
    head = head_list[-1]
    body = tuple(body.split(", "))  # Convert body to tuple for hashability
    return score, head, body


def load_rules(rule_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_filepath in glob.glob(os.path.join(rule_path, "*.txt")):
        with open(input_filepath, 'r') as f:
            lines = f.readlines()

        exist_rules = set()  # To track unique rules based on their bodies
        processed_rules = []

        for i_, rule in enumerate(lines):
            try:
                score, head, body = parse_rule(rule.strip('\n'))

                if body in exist_rules:  # Skip duplicate rules
                    continue

                parts = rule.strip().split('\t')
                if len(parts) != 5:
                    print(f"Skipping malformed line: {rule}")
                    continue

                support, coverage, confidence, pca_confidence = map(float, parts[:4])

                if support == 0:
                    continue
                exist_rules.add(body)

                rule_info = parts[4]

                processed_rules.append({
                    'support': support,
                    'coverage': coverage,
                    'confidence': confidence,
                    'pca_confidence': pca_confidence,
                    'rule_info': rule_info,
                    'original_line': rule,  # Keep the original line for reference
                })
            except ValueError as e:
                print(f"Error processing line {i_}: {rule}. Error: {e}")
                continue

        # Sort by pca_confidence in descending order
        sorted_rules = sorted(processed_rules, key=lambda x: x['pca_confidence'], reverse=True)

        # Write sorted data back to file
        output_file_path = os.path.join(output_folder, f"{head}_sorted_rules.txt")
        with open(output_file_path, 'w') as f:
            for rule in sorted_rules:
                f.write(rule['original_line'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='merge_rules')
    parser.add_argument('--sorted_rule_path', default='sorted_rules')
    parser.add_argument("--dataset", default="family")
    parser.add_argument("-p", default='gpt-3.5-turbo-top-0-f-50-l-10', help="rule path")
    # parser.add_argument("-t", default="0205", help="The date on which the code was run")
    args = parser.parse_args()

    input_folder = os.path.join(args.input_folder, args.dataset, args.p)
    # input_folder = os.path.join(args.input_folder, args.dataset)
    output_folder = os.path.join(args.sorted_rule_path, args.dataset, args.p)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    load_rules(input_folder, output_folder)
