# Extending-Healthbench
## Research Overview
Exttending HealthBench is a research project focused on improving the safety, accuracy, and cultural relevance of medical AI systems for African communities.

Today, nearly all medical evaluation datasets (including the original OpenAI HealthBench) exist only in English. This results in:

- Models producing unsafe or incorrect medical outputs when used in African languages
- Lack of evaluation tools for African languages
- Inability for researchers to measure how well LLMs perform in real local contexts
  
Our project addresses these gaps by building the first multilingual medical AI evaluation benchmark for African languages, starting with:
- Igbo
- Yoruba
- Nigerian Pidgin
- Kikuyu
- Swahili
  
We combine LLM-as-judge evaluations and human medical expert evaluations to measure safety, correctness, and usability across languages.
## Why this matters
Medical AI is becoming increasingly influential yet African languages remain underrepresented. This causes:
- Misinformation in medical conversations
- Wrong dosage instructions
- Misunderstandings in lower-literacy communities
- Reduced trust in AI health tools

By building a multilingual benchmark, we enable:
- Safer, locally-relevant medical AI systems
- Better tools for African NLP researchers
- Evaluation methods tailored to real-world African health contexts
- A baseline for future medical AI development in African languages

## Project Outputs
- A multilingual translation of the OpenAI HealthBench dataset
- An evaluation benchmark
- A research paper

## Project Code Structure
```
Tonative-healthbench/
│
|── data/
│   ├── (local_datasets).jsonl             # original dataset
|
|── output_data/
│   ├── (translated_dataset_for_openai).jsonl             # translated dataset
|   ├── (translated_dataset_for_openai).jsonl      
|
├── translator/
│
│   ├── healthbench_translator.py       # main translation pipeline
│   ├── providers/
│   │   ├── openai_provider.py          # OpenAI wrapper
│   │   ├── claude_provider.py          # Anthropic wrapper
│   │
│   ├── prompts/                        # per-language translation rules
│   │   ├── yoruba.json
│   │   ├── igbo.json
│   │   ├── pidgin.json
│   │   ├── kikuyu.json
│   │   ├── swahili.json
│   │
│   ├── artifacts/
│   │   ├── artifacts_patterns.py           # removes artifacts from translations
│      
├── scripts/
│   ├── run_translation.py              # CLI runner
│   ├── run_batch_job.py                # batch job status + download
│
├── .env.example
├── requirements.txt
└── README.md

```
## Installation
- Clone the project
```
git clone <repo-url>
cd <repo-name>
```
- Create virtual environment
```
python -m venv venv
venv\Scripts\activate   # Windows
```
- Install dependencies
```
pip install -r requirements.txt
```
- Environment variables
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_claude_key
```
## Running a translation
- For OpenAI run
```
 python scripts/run_translation.py --provider openai --model gpt-4o-mini --lang {language e.g igbo} --input data/{dataset} --output output_data/{language_batchnumber_LLm}.jsonl --test_size {number}
```
- For Claude
```
 python scripts/run_translation.py --provider claude --model claude-3-haiku-20240307 --lang {language e.g igbo} --input data/{dataset}.jsonl --output output_data/{language_batchnumber_LMM}.jsonl --test_size {number}
```
## Things to note
- In the artifacts_patterns.py, you would have to include the version for the language you are translating to. an example for the Igbo language is shown below
```
  patterns  = [
            r'\b(Unchanged|No change|Keep in English|As stated|Maintain|Same as before)[:]*\s*',
            r'\b(Class remains|Status remains|Value remains)[:]*\s*',
            r'\b(Translation note|Note|Comment)[:]*\s*',

            # Igbo versions
            r'\b(Ogbanweghị mgbanwe|Enweghị mgbanwe|Ka ọ dị)[:]*\s*',
            r'\b(Ka ọ dị tupu|O yiri nke mbụ)[:]*\s*',

            r'\s*\([^)]*unchanged[^)]*\)',
            r'\s*\([^)]*same[^)]*\)',

            r'\s*-\s*(unchanged|no change|same)\s*',
            r'\s*–\s*(unchanged|no change|same)\s*',
        ]
```
- In the prompts folder, create a new prompt file for the language you are translating to example: ```igbo.json```, then add this lines of code changing every occurance of igbo to your target language
```
  {
  "language": "igbo",
  "system_prompt": "You are an expert translator specializing in English → Igbo medical translation.\n\nTranslation rules:\n1. Keep drug names in English.\n2. Keep medical abbreviations like CPR, ECG, AHA in English.\n3. Keep measurements as digits.\n4. Translate descriptive text naturally.\n5. Keep academic citations exactly.\n6. Output ONLY the Igbo translation.\n\nTranslate the following medical text into Igbo:"}
```
- In the healthbench_translator.py file, change every occurance of igbo to your target language. Below are the parts of the file where you have to make the changes.
1. 
```
def __init__(
        self,
        provider: str,
        api_key: str,
        lang: str = "igbo",
        model: Optional[str] = None,
        use_batch: bool = True,
        prompts_dir: str = "translator/prompts"
    )
```
Change ```lang: str = "igbo",``` to your target language














