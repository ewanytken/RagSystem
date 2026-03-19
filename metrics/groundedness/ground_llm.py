import json
import re
from typing import List, Dict, Optional

import numpy as np

from app.respondent.abstract_respondent import Respondent
from app.utils import Utils
from metrics.groundedness.ground_base import GroundednessResult, GroundednessScore
from test_metrics import logger_metrics


class LLMGroundedness:

    def __init__(self, model: Optional[Respondent], config: Dict):
        self.model = model
        self.config = config

    def create_groundedness_prompt(self,
                                   answer: str,
                                   contexts: List[str]) -> str:
        context = "\n\n".join([
            f"[Document {i + 1}]: {ctx[:500]}"
            for i, ctx in enumerate(contexts[:5])
        ])

        template = self.config['prompts']['groundedness']
        prompt = template.format(answer=answer, context=context)
        return prompt

    def parse_llm_response(self, response: str) -> Dict:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_parse(response)
        except Exception as e:
            logger_metrics(f"LLM response parsing ERROR {e}")

    def _fallback_parse(self, response: str) -> Dict:
        lines = response.strip().split('\n')
        claims = []

        for line in lines:
            if 'claim' in line.lower() and 'supported' in line.lower():
                claims.append({
                    "text": line,
                    "supported": 'true' in line.lower(),
                    "confidence": 0.5
                })

        return {
            "claims": claims,
            "summary": {
                "total_claims": len(claims),
                "supported_claims": sum(1 for c in claims if c.get("supported", False)),
                "groundedness_score": len([c for c in claims if c.get("supported", False)]) / max(len(claims), 1)
            }
        }

    def evaluate(self,
                 response: str,
                 context: List[str],
                 return_details: bool = True) -> GroundednessResult:

        prompt = self.create_groundedness_prompt(response, context)
        response = self.model.generate(prompt)

        summary = response.get("summary", {})
        claims_data = response.get("claims", [])

        score = summary.get("groundedness_score", 0)
        total_claims = summary.get("total_claims", len(claims_data))
        supported = summary.get("supported_claims",
                                sum(1 for c in claims_data if c.get("supported", False)))

        # Categorize claims
        supported_claims = [
            c["text"] for c in claims_data
            if c.get("supported", False) and "text" in c
        ]
        unsupported_claims = [
            c["text"] for c in claims_data
            if not c.get("supported", True) and "text" in c
        ]
        if score >= 0.8:
            level = GroundednessScore.HIGH
        elif score >= 0.5:
            level = GroundednessScore.MEDIUM
        elif score >= 0.2:
            level = GroundednessScore.LOW
        else:
            level = GroundednessScore.NONE

        confidences = [c.get("confidence", 0.5) for c in claims_data if "confidence" in c]
        avg_confidence = np.mean(confidences) if confidences else 0.8

        return GroundednessResult(
            score=score,
            level=level,
            total_claims=total_claims,
            supported_claims=supported,
            unsupported_claims=unsupported_claims,
            supported_claims_list=supported_claims,
            details={
                "method": "llm_based",
                "model": self.model,
                "raw_response": response if return_details else None,
                "claim_details": claims_data if return_details else None
            },
            confidence=avg_confidence
        )