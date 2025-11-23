import re

class ArtifactsPatterns:
    def clean(self, text: str) -> str:
        if not text:
            return text
        
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
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'\s*:\s*:', ':', cleaned)
            cleaned = re.sub(r'\s*,\s*,', ',', cleaned)
            cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
            
            return cleaned.strip()
        
        
        
        