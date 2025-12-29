# LLMGuardrail 🛡️

Open-source guardrail toolkit for Large Language Models.

Protects against **prompt injection** and **jailbreak attacks** — OWASP LLM01:2025 #1 risk.

**Features**
- Hybrid detection (SOTA models + heuristics)
- Adversarial testing suite
- MLOps, DevSecOps, Data Engineering
- FastAPI service + Streamlit dashboard
- Integrations: LangChain, LlamaIndex

Built for CC7016NI coursework — free forever (MIT license).

**Quick Start**
```bash
pip install llm-guardrail
guardrail detect "your prompt here"