import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from dotenv import load_dotenv
load_dotenv()

import argparse
from translator.healthbench_translator import HealthBenchTranslator


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--provider", required=True, help="openai or claude")
    parser.add_argument("--model", default=None)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--test_size", type=int)
    
    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY") if args.provider == "openai" else os.getenv("ANTHROPIC_API_KEY")
    
    translator = HealthBenchTranslator(
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        lang=args.lang
    )
    
    translator.translate_dataset(
        input_file=args.input,
        output_file=args.output,
        test_size=args.test_size
    )
    
if __name__ == "__main__":
    main()