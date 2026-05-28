import re

class EnterpriseGuardrails:
    """
    Implements production-grade Input/Output Guardrails.
    Protects against prompt injections, system override attempts, jailbreaks,
    PII leakages, and standard safety policy violations.
    """
    def __init__(self):
        # Compiled patterns for optimal efficiency
        self.injection_patterns = [
            re.compile(r"ignore previous instructions", re.IGNORECASE),
            re.compile(r"system override", re.IGNORECASE),
            re.compile(r"you are now a", re.IGNORECASE),
            re.compile(r"bypass security", re.IGNORECASE),
            re.compile(r"jailbreak", re.IGNORECASE),
            re.compile(r"forget your settings", re.IGNORECASE),
            re.compile(r"sudo\b", re.IGNORECASE),
            re.compile(r"rm -rf", re.IGNORECASE),
            re.compile(r"<script>", re.IGNORECASE)
        ]
        
        self.pii_patterns = {
            "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
            "us_ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        }
        
        self.forbidden_topics = [
            "illegal weapons",
            "malware source code",
            "unauthorized hacking"
        ]

    def validate_input(self, query: str) -> dict:
        """
        Inspects query for injections, hacks, or system abuses.
        Returns a dict tracking status and safety reasons.
        """
        # 1. Prompt Injection Scanning
        for pattern in self.injection_patterns:
            if pattern.search(query):
                return {
                    "is_safe": False,
                    "reason": "PROMPT_INJECTION_DETECTED",
                    "details": f"Query triggered pattern: {pattern.pattern}"
                }
                
        # 2. Forbidden Topics Check
        query_lower = query.lower()
        for topic in self.forbidden_topics:
            if topic in query_lower:
                return {
                    "is_safe": False,
                    "reason": "FORBIDDEN_TOPIC_VIOLATION",
                    "details": f"Query discussed blacklisted topic: {topic}"
                }
                
        return {
            "is_safe": True,
            "reason": None,
            "details": "Query passed all input guardrail checks."
        }

    def validate_output(self, response: str) -> dict:
        """
        Inspects output response for PII leakage, leaks of API keys,
        or forbidden text segments before delivery.
        """
        # 1. PII Scan
        for pii_type, pattern in self.pii_patterns.items():
            if pattern.search(response):
                # Clean or redact the response, or flag it
                redacted_response = pattern.sub("[REDACTED]", response)
                return {
                    "is_safe": False,
                    "reason": "PII_LEAK_DETECTED",
                    "details": f"Response contained PII elements of type: {pii_type}",
                    "repaired_response": redacted_response
                }
                
        # 2. Key Leaks Scan (e.g. AWS Keys or Gemini Keys)
        if re.search(r"AIzaSy[A-Za-z0-9-_]{35}", response) or re.search(r"sk-[A-Za-z0-9]{48}", response):
            return {
                "is_safe": False,
                "reason": "SECRET_KEY_LEAK_DETECTED",
                "details": "Response leaked active API/Secret keys.",
                "repaired_response": "[REDACTED SECURITY ALERT: Secret token output blocked.]"
            }
            
        return {
            "is_safe": True,
            "reason": None,
            "details": "Response passed all output guardrail checks.",
            "repaired_response": response
        }
