import os
import json
import argparse

from scorers.pattern_scorer import PatternScorer, PatternScorerConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scorer', type=str, default='PatternScorer', help='Scorer name')
    parser.add_argument('--config_path', type=str, default='./config/pattern_scorer.yaml', help='Path to config file')
    parser.add_argument('--json_path', type=str, required=True, help='Path to json file for evaluation')
    parser.add_argument('--output_path', type=str, default='eval_summary.txt', help='Path to output summary txt')
    args = parser.parse_args()

    scorer_name = args.scorer
    json_path = args.json_path
    output_path = args.output_path
    config_path = args.config_path

    scorer_class = globals().get(scorer_name, None)
    config_cls = globals().get(f"{scorer_name}Config", None)
    scorer_instance = None
    if scorer_class is not None:
        scorer_instance = scorer_class(config_cls.from_yaml(config_path))
    else:
        raise ValueError(f"Scorer {scorer_name} is not defined or does not have a score method.")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(output_path)
    base_filename = base_filename.split('.')[0] + f'_{scorer_name}.' + base_filename.split('.')[-1]
    output_path = os.path.join(output_dir, base_filename)

    # Read input json
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_samples = 0
    attack_success_count = 0

    for line in lines:
        item = json.loads(line.strip())
        total_samples += 1
        query = None
        response = None
        query_field_candidates = ['goals', 'query', 'final_query', 'input']
        response_field_candidates = ['response', 'target', 'targets']
        for field in query_field_candidates:
            if field in item:
                query = item[field]
                break
        for field in response_field_candidates:
            if field in item:
                response = item[field]
                break
        if isinstance(response, list) and len(response) == 1:
            response = response[0]
        assert isinstance(query, str) and isinstance(response, str), "Query and response must be strings"
        if isinstance(query, str) and isinstance(response, str):
            score = scorer_instance.score(query, response)
            if score['score']:
                attack_success_count += 1
        
        

    asr = attack_success_count / total_samples if total_samples > 0 else 0.0

    summary_lines = [
        f"Evaluation Summary Report",
        f"Scorer: {scorer_name}",
        f"Input file: {json_path}",
        f"Total samples: {total_samples}",
        f"Attack success samples: {attack_success_count}",
        f"Attack Success Rate (ASR): {asr:.4f}",
        "",
    ]
    summary_text = "\n".join(summary_lines)

    print(summary_text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + "\n")

if __name__ == "__main__":
    main()
