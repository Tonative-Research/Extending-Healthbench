import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from dotenv import load_dotenv
load_dotenv()

from translator.healthbench_translator import HealthBenchTranslator


translator = HealthBenchTranslator(
    provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    lang="igbo"
)

status = translator.check_batch_status("batch_691c3e358c90819097f4b0b416ada73a")
print(status)

translator.download_and_save_results(
    batch_id="batch_691c3e358c90819097f4b0b416ada73a",
    original_input_file="data/2025-05-07-06-14-12_oss_eval.jsonl",
    output_file="output_data/igbo_test50.jsonl"
)
