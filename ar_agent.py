"""AR agent entrypoint for the Opik Agent Playground.

Runs as a connected runner: `opik endpoint --headless --project ar-voice-agent -- python3 ar_agent.py`
Uses the local vLLM model (OpenAI-compatible).
"""
from openai import OpenAI

import opik

client = OpenAI(base_url="http://vllm:8001/v1", api_key="local")
MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


@opik.track(entrypoint=True, project_name="ar-voice-agent")
def ar_agent(query: str, claim_id: str = "unknown") -> str:
    """AR specialist: answer a claim question using the local LLM."""
    system = (
        "You are an AR (Accounts Receivable) specialist calling insurance companies. "
        "Be concise and direct. Do not repeat yourself."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Claim {claim_id}. {query}"},
        ],
        max_tokens=200,
        temperature=0,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    # Keep the runner process alive so the Agent Playground can submit jobs.
    import time
    while True:
        time.sleep(3600)
