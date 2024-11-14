from langcheck import get_report
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--datasetpath',
                        type=str,
                        required=True,
                        help='A path to .parquet dataset')
    parser.add_argument('--path4report',
                        type=str,
                        required=True,
                        help='A name for JSON file')
    parser.add_argument('--LLMscoring',
                        type=bool,
                        default=True)
    parser.add_argument('--llm',
                        type=str,
                        help='Result file',
                        default='mistralai/Mistral-7B-Instruct-v0.2')

    args = parser.parse_args()

    report = get_report(path=args.datasetpath,
                        ask_llm=args.LLMscoring,
                        llm2score=args.llm
                        )

    with open(f'reports/{args.path4report}', 'w') as wp:
        json.dump(report, wp)