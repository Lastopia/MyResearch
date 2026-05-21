import json
import urllib.error
import urllib.request

from para import SECRETS


class FeatureInterpreter:
    def __init__(self, interp_cfg):
        self.interp_cfg = interp_cfg

    def prompt_for_feature(self, item):
        contexts = "\n".join(
            [
                (
                    f"{idx + 1}. activated_token={ctx['activated_token']!r}, "
                    f"activation={ctx['activation']:.4f}, position={ctx['position']}, "
                    f"context={ctx['context']!r}"
                )
                for idx, ctx in enumerate(item["contexts"])
            ]
        )
        return f"""You are evaluating a sparse autoencoder feature from a transformer.

The model identity and position encoding are blinded. Use only the activation examples.

Return strict JSON with these keys:
feature_type: one of ["content", "position", "mixed", "low-level", "undiscernible"]
interpretability_score: integer 1-5
specificity_score: integer 1-5
coverage_score: integer 1-5
false_positive_risk: integer 1-5
confidence_score: number 0-1
short_explanation: string
evidence_summary: string

Rubric:
5 means a clear, consistent pattern across almost all examples.
3 means a plausible but incomplete pattern.
1 means no stable pattern.
false_positive_risk is higher when the explanation is overly broad.
confidence_score is your confidence that the explanation is supported by the shown contexts.

Feature metadata:
blinded_feature_id: {item['blinded_feature_id']}
layer: {item['layer']}
phase5_label: {item['phase5_label']}

Top activating contexts:
{contexts}
"""

    def dry_run_response(self, item):
        label = item["phase5_label"]
        if label == "position_only":
            feature_type = "position"
            explanation = "Dry run: feature was selected as position-related by Phase 5 scores."
        elif label == "content_only":
            feature_type = "content"
            explanation = "Dry run: feature was selected as content-related by Phase 5 scores."
        elif label == "mixed":
            feature_type = "mixed"
            explanation = "Dry run: feature was selected as mixed by Phase 5 scores."
        elif len(item["contexts"]) < getattr(self.interp_cfg, "min_active_contexts", 4):
            feature_type = "undiscernible"
            explanation = "Dry run: too few active contexts for a reliable interpretation."
        else:
            feature_type = "undiscernible"
            explanation = "Dry run placeholder; set INTERP.dry_run=False to call OpenAI."
        return {
            "feature_type": feature_type,
            "interpretability_score": 3,
            "specificity_score": 3,
            "coverage_score": 3,
            "false_positive_risk": 3,
            "confidence_score": 0.5,
            "short_explanation": explanation,
            "evidence_summary": "Dry run generated no external LLM evidence.",
            "raw_response": None,
        }

    def call_openai(self, prompt):
        api_key = getattr(SECRETS, "openai_api_key", None)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        body = {
            "model": getattr(self.interp_cfg, "model", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "developer",
                    "content": "You are a careful mechanistic interpretability evaluator. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": getattr(self.interp_cfg, "temperature", 0.0),
            "max_tokens": getattr(self.interp_cfg, "max_tokens", 700),
            "response_format": {"type": "json_object"},
            "store": False,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        parsed["raw_response"] = content
        parsed["raw_api_response"] = payload
        return parsed

    def score_item(self, item):
        prompt = self.prompt_for_feature(item)
        item["prompt"] = prompt
        if getattr(self.interp_cfg, "dry_run", True):
            return self.dry_run_response(item)
        return self.call_openai(prompt)

    def quality_score(self, response):
        interp = float(response.get("interpretability_score", 0))
        spec = float(response.get("specificity_score", 0))
        cov = float(response.get("coverage_score", 0))
        risk = float(response.get("false_positive_risk", 5))
        return ((interp + spec + cov) / 3.0) - 0.25 * (risk - 1.0)
