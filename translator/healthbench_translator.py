import json
import time
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

# Provider wrappers and cleaners
from translator.providers.openai_provider import OpenAIProvider
from translator.providers.claude_provider import ClaudeProvider
from translator.artifacts.artifacts_patterns import ArtifactsPatterns


class HealthBenchTranslator:
    """
    Generalized translator for HealthBench dataset.
    Supports multiple languages and providers (OpenAI, Claude).
    Keeps most original behaviour (batch via OpenAI, realtime for all).
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        lang: str = "igbo",
        model: Optional[str] = None,
        use_batch: bool = True,
        prompts_dir: str = "translator/prompts"
    ):
        """
        Args:
            provider: "openai" or "claude"
            api_key: API key for chosen provider
            lang: target language (e.g., "yoruba","igbo","pidgin","kikuyu")
            model: model name (if None, defaults: openai -> "gpt-4o-mini", claude -> "claude-3-sonnet")
            use_batch: allow using OpenAI batch API when available
            prompts_dir: path to JSON prompts folder
        """

        self.provider_name = provider.lower()
        if self.provider_name not in ("openai", "claude"):
            raise ValueError("provider must be 'openai' or 'claude'")

        self.api_key = api_key
        self.lang = lang.lower()
        self.use_batch = use_batch
        self.prompts_dir = prompts_dir

        # Default models if not provided
        if model:
            self.model = model
        else:
            self.model = "gpt-4o-mini" if self.provider_name == "openai" else "claude-3-sonnet"

        # Initialize provider wrapper
        if self.provider_name == "openai":
            self.provider = OpenAIProvider(api_key=self.api_key, model=self.model)
        else:
            self.provider = ClaudeProvider(api_key=self.api_key, model=self.model)

        self.cleaner = ArtifactsPatterns()

        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'translation_{self.lang}_{self.provider_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Load language prompt
        self.system_prompt = self._load_language_prompt(self.lang)


    def _load_language_prompt(self, lang: str) -> str:
        """
        Load the JSON prompt for the requested language.
        Expect file at: {prompts}/{igbo}.json
        If not present, use a generic fallback prompt.
        """
        prompt_path = os.path.join(self.prompts_dir, f"{lang}.json")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                sp = cfg.get("system_prompt") or cfg.get("prompt") or cfg.get("system") or ""
                if not sp:
                    raise ValueError("prompt file exists but 'system_prompt' missing")
                return sp
        except FileNotFoundError:
            self.logger.warning(f"No prompt JSON for language '{lang}' at {prompt_path}. Using fallback prompt.")
            return (
                f"You are an expert medical translator specializing in English to {lang.capitalize()} translation.\n\n"
                "Guidelines:\n"
                " - Keep drug names, abbreviations, measurements and study names in English.\n"
                " - Translate descriptive medical text naturally into the target language.\n"
                " - Maintain professional tone and formatting.\n"
                " - Do NOT add meta-comments or 'unchanged' markers.\n\n"
                f"Translate the following medical text into {lang.capitalize()}:"
            )
        except Exception as e:
            self.logger.error(f"Failed to load prompt for {lang}: {e}")
            raise

   
    def clean_translation_artifacts(self, text: str) -> str:
        """
        Wrapper that uses BaseCleaner to remove artifacts and also applies
        any lightweight language-specific regexes if needed.
        """
        cleaned = self.cleaner.clean(text)

        # Additional lightweight language-specific removals (extendable)
        if self.lang == "igbo":
            # common Igbo artifact patterns (examples)
            igbo_patterns = [
                r'\b(Enweghị mgbanwe|Ka ọ dị)[:]*\s*',
                r'\b(O yiri nke mbụ|O ka dị otu a)[:]*\s*'
            ]
            for p in igbo_patterns:
                cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # if self.lang == "pidgin":
        #     pidgin_patterns = [
        #         r'\b(E remain same|Same same|No change dey)[:]*\s*'
        #     ]
        #     for p in pidgin_patterns:
        #         cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE)
        #     cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Other languages can be appended similarly
        return cleaned.strip() if cleaned else cleaned

   
    def translate_text(self, text: str) -> Optional[str]:
        """
        Translate a single chunk of text using the configured provider (realtime).
        Returns cleaned translation or None on failure.
        """
        if not text or not text.strip():
            return None

        try:
            raw = self.provider.translate(self.system_prompt, text)
            if raw is None:
                self.logger.warning("Provider returned None for translation.")
                return None
            cleaned = self.clean_translation_artifacts(raw.strip())
            return cleaned
        except Exception as e:
            self.logger.error(f"translate_text() failed for input prefix '{text[:60]}...': {e}")
            return None

   
    def create_batch_request(self, texts_to_translate: List[Dict]) -> str:
        """
        Create JSONL for OpenAI batch API.
        Each item in texts_to_translate should be dict {id, text, field_type}
        Returns path to created batch file.
        """
        if self.provider_name != "openai":
            raise RuntimeError("Batch requests are implemented only for OpenAI provider in this script.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = f"batch_translation_request_{self.lang}_{timestamp}.jsonl"

        with open(batch_file, "w", encoding="utf-8") as f:
            for item in texts_to_translate:
                body = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": item["text"]}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4000
                }
                req = {
                    "custom_id": item["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

        self.logger.info(f"Created batch request file: {batch_file} ({len(texts_to_translate)} requests)")
        return batch_file

    def submit_batch_job(self, batch_file_path: str) -> str:
        """
        Submit the batch file to OpenAI via provider client.
        Returns batch_job.id
        """
        if self.provider_name != "openai":
            raise RuntimeError("Batch submission implemented for OpenAI only in this script.")

        try:
            # Upload file
            with open(batch_file_path, "rb") as f:
                upload = self.provider.client.files.create(file=f, purpose="batch")
            input_file_id = upload.id

            # Create batch
            batch_job = self.provider.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"HealthBench {self.lang} translation"}
            )
            self.logger.info(f"Submitted batch job {batch_job.id}")
            return batch_job.id
        except Exception as e:
            self.logger.error(f"submit_batch_job failed: {e}")
            raise

    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Retrieve status for an OpenAI batch job.
        """
        if self.provider_name != "openai":
            return {"status": "unsupported", "error": "batch unsupported for non-OpenAI provider"}

        try:
            batch = self.provider.client.batches.retrieve(batch_id)
            info = {
                "status": getattr(batch, "status", None),
                "completed_at": getattr(batch, "completed_at", None),
                "failed_at": getattr(batch, "failed_at", None),
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "request_counts": getattr(batch, "request_counts", None)
            }
            return info
        except Exception as e:
            self.logger.error(f"check_batch_status error: {e}")
            return {"status": "error", "error": str(e)}

    def download_batch_results(self, batch_id: str) -> Dict[str, Optional[str]]:
        """
        Download the batch output file and parse items into {custom_id: cleaned_text}
        (OpenAI batch only).
        """
        if self.provider_name != "openai":
            raise RuntimeError("Batch download only implemented for OpenAI provider here.")

        info = self.check_batch_status(batch_id)
        status = info.get("status")
        if status != "completed":
            raise RuntimeError(f"Batch not completed. Status: {status}")

        output_file_id = info.get("output_file_id")
        if not output_file_id:
            raise RuntimeError("No output_file_id on completed batch.")

        # Download file content
        file_obj = self.provider.client.files.content(output_file_id)
        text = file_obj.text

        results: Dict[str, Optional[str]] = {}
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id")
            resp = item.get("response", {})
            status_code = resp.get("status_code")
            if status_code == 200:
                body = resp.get("body", {})
                choices = body.get("choices", [])
                if choices and choices[0].get("message"):
                    raw_translation = choices[0]["message"]["content"]
                    results[custom_id] = self.clean_translation_artifacts(raw_translation.strip())
                else:
                    results[custom_id] = None
                    self.logger.warning(f"No choices for {custom_id} in batch output.")
            else:
                results[custom_id] = None
                self.logger.error(f"Batch item failed for {custom_id}: {item}")
        self.logger.info(f"Downloaded and parsed {len(results)} batch results.")
        return results


    def prepare_translation_data(self, data: List[Dict]) -> List[Dict]:
        """
        Convert HealthBench records into flat list of translation requests.
        Each request is a dict with keys: id, text, field_type, record_index, (maybe ref_index)
        """
        translation_requests: List[Dict] = []

        for i, record in enumerate(data):
            record_id = record.get("prompt_id", f"record_{i}")

            # 1. Prompt text
            if "prompt" in record and record["prompt"] and len(record["prompt"]) > 0:
                prompt_content = record["prompt"][0].get("content", "")
                if prompt_content and prompt_content.strip():
                    translation_requests.append({
                        "id": f"{record_id}_prompt",
                        "text": prompt_content,
                        "field_type": "prompt",
                        "record_index": i
                    })

            # 2. Ideal completion
            if "ideal_completions_data" in record and record["ideal_completions_data"]:
                ideal = record["ideal_completions_data"].get("ideal_completion", "")
                if ideal and ideal.strip():
                    translation_requests.append({
                        "id": f"{record_id}_ideal",
                        "text": ideal,
                        "field_type": "ideal_completion",
                        "record_index": i
                    })

                # 3. Reference completions
                refs = record["ideal_completions_data"].get("ideal_completions_ref_completions", [])
                if refs:
                    for j, ref in enumerate(refs):
                        if ref and ref.strip():
                            translation_requests.append({
                                "id": f"{record_id}_ref_{j}",
                                "text": ref,
                                "field_type": "ref_completion",
                                "record_index": i,
                                "ref_index": j
                            })
        self.logger.info(f"Prepared {len(translation_requests)} translation requests from {len(data)} records.")
        return translation_requests

    
    def translate_dataset(self, input_file: str, output_file: Optional[str] = None, test_size: Optional[int] = None) -> Optional[str]:
        """
        Main entrypoint: load dataset, prepare requests, translate (batch or realtime), save output.
        Returns output_file path or BATCH_JOB:<id> string when batch submitted.
        """
        # Load original dataset
        self.logger.info(f"Loading data from {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        if test_size:
            data = data[:test_size]
            self.logger.info(f"Using test size: {test_size}")

        if not output_file:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_test{test_size}" if test_size else ""
            output_file = f"healthbench_{self.lang}_{self.provider_name}{suffix}_{ts}.jsonl"

        # Prepare flat requests
        requests = self.prepare_translation_data(data)
        if not requests:
            self.logger.error("No translation requests prepared; exiting.")
            return None

        # If using batch and provider is openai, create + submit batch
        if self.use_batch and self.provider_name == "openai" and len(requests) > 10:
            self.logger.info("Preparing OpenAI batch job...")
            batch_file = self.create_batch_request(requests)
            batch_id = self.submit_batch_job(batch_file)
            self.logger.info(f"Batch submitted: {batch_id}")
            self.logger.info("When batch completes, call download_and_save_results(batch_id, input_file, output_file)")
            return f"BATCH_JOB:{batch_id}"

        # Otherwise use realtime translation
        translations: Dict[str, Optional[str]] = {}
        total = len(requests)
        self.logger.info(f"Starting realtime translation for {total} items (provider={self.provider_name})")
        for idx, req in enumerate(requests):
            self.logger.info(f"({idx+1}/{total}) Translating {req['field_type']} id={req['id']}")
            translated = self.translate_text(req["text"])
            translations[req["id"]] = translated
            # small pause to avoid hitting enforced rate limits
            time.sleep(0.1)

        # Save merged dataset
        self.save_translated_dataset(data, translations, output_file)
        return output_file

   
    def save_translated_dataset(self, original_data: List[Dict], translations: Dict[str, Optional[str]], output_file: str):
        """
        Merge translations into original records and save JSONL.
        """
        self.logger.info(f"Saving translated dataset to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for i, record in enumerate(original_data):
                record_id = record.get("prompt_id", f"record_{i}")
                translated_record = record.copy()

                translated_record["translation_info"] = {
                    "translated_at": datetime.now().isoformat(),
                    "lang": self.lang,
                    "provider": self.provider_name,
                    "model": self.model
                }

                # Prompt
                p_key = f"{record_id}_prompt"
                if p_key in translations and translations[p_key]:
                    translated_record.setdefault("prompt", [{}])
                    translated_record["prompt"][0]["translated_content"] = translations[p_key]

                # Ideal
                i_key = f"{record_id}_ideal"
                if i_key in translations and translations[i_key]:
                    translated_record.setdefault("ideal_completions_data", {})
                    translated_record["ideal_completions_data"]["translated_ideal_completion"] = translations[i_key]

                # Reference completions
                refs = (translated_record.get("ideal_completions_data") or {}).get("ideal_completions_ref_completions", [])
                translated_refs = []
                for j in range(len(refs)):
                    r_key = f"{record_id}_ref_{j}"
                    translated_refs.append(translations.get(r_key))
                # always set the translated refs field (preserve structure)
                if not isinstance(translated_record.get("ideal_completions_data"), dict):
                    translated_record["ideal_completions_data"] = {}
                translated_record["ideal_completions_data"]["translated_ref_completions"] = translated_refs

                f.write(json.dumps(translated_record, ensure_ascii=False) + "\n")

        self.logger.info(f"Saved {len(original_data)} translated records to {output_file}")

    
    def download_and_save_results(self, batch_id: str, original_input_file: str, output_file: Optional[str] = None):
        """
        For OpenAI batch jobs: download results, merge into original dataset and save.
        """
        if self.provider_name != "openai":
            raise RuntimeError("download_and_save_results implemented only for OpenAI batch in this script.")

        # load original data
        self.logger.info("Loading original data for merging")
        with open(original_input_file, "r", encoding="utf-8") as f:
            original_data = [json.loads(line) for line in f if line.strip()]

        # determine output file name if not provided
        if not output_file:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"healthbench_{self.lang}_{self.provider_name}_batch_{ts}.jsonl"

        # download translations
        translations = self.download_batch_results(batch_id)

        # save merged dataset
        self.save_translated_dataset(original_data, translations, output_file)
        self.logger.info(f"Batch translations merged and saved to {output_file}")
        return output_file
