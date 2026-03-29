"""
Multi-Agent Loop: GLM-4.7 Coder + Reviewer
"""

import os
from openai import OpenAI

BASE_URL = "https://api.z.ai/api/paas/v4"
MODEL = "glm-4.7"
MAX_ROUNDS = 5

client = OpenAI(
    api_key=os.environ.get("GLM_API_KEY", "your-api-key-here"),
    base_url=BASE_URL,
)


def call_agent(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_code(text: str) -> str:
    """Extract code from markdown code block if present."""
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # inside code block
                # remove language tag (e.g. "python\n")
                lines = part.split("\n")
                if lines[0].strip().lower() in ("python", "py", ""):
                    return "\n".join(lines[1:]).strip()
                return part.strip()
    return text.strip()


CODER_SYSTEM = """You are an expert Python developer.
Your job is to write clean, correct, and well-structured Python code based on the given requirement.
If you receive feedback from a reviewer, revise the code to address all issues.
Always respond with ONLY the code inside a ```python ... ``` block — no extra explanation outside the block."""

REVIEWER_SYSTEM = """You are a senior code reviewer.
Your job is to review Python code for correctness, readability, edge cases, and best practices.
Respond with one of the following:
- If the code is good: start your response with "APPROVED" and briefly explain why.
- If the code needs improvement: start your response with "REVISION NEEDED" and list specific issues clearly."""


def run_agent_loop(requirement: str, output_file: str):
    print(f"\n{'='*60}")
    print(f"Requirement: {requirement}")
    print(f"Output file: {output_file}")
    print(f"Max rounds : {MAX_ROUNDS}")
    print(f"{'='*60}\n")

    code = ""
    reviewer_feedback = ""

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"--- Round {round_num}/{MAX_ROUNDS} ---")

        # --- Coder ---
        if round_num == 1:
            coder_prompt = f"Write Python code for the following requirement:\n\n{requirement}"
        else:
            coder_prompt = (
                f"Requirement:\n{requirement}\n\n"
                f"Previous code:\n```python\n{code}\n```\n\n"
                f"Reviewer feedback:\n{reviewer_feedback}\n\n"
                "Please revise the code to fix all issues mentioned above."
            )

        print("[Coder] Writing code...")
        coder_response = call_agent(CODER_SYSTEM, coder_prompt)
        code = extract_code(coder_response)
        print(f"[Coder] Done. ({len(code.splitlines())} lines)\n")

        # --- Reviewer ---
        reviewer_prompt = (
            f"Requirement:\n{requirement}\n\n"
            f"Code to review:\n```python\n{code}\n```"
        )

        print("[Reviewer] Reviewing code...")
        reviewer_response = call_agent(REVIEWER_SYSTEM, reviewer_prompt)
        reviewer_feedback = reviewer_response
        approved = reviewer_response.strip().upper().startswith("APPROVED")

        print(f"[Reviewer] Verdict: {'APPROVED' if approved else 'REVISION NEEDED'}")
        print(f"[Reviewer] Feedback:\n{reviewer_response}\n")

        if approved:
            print(f"Code approved in round {round_num}!")
            break
        elif round_num == MAX_ROUNDS:
            print(f"Max rounds ({MAX_ROUNDS}) reached. Saving best code so far.")

    # --- Save output ---
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"\n{'='*60}")
    print(f"Final code saved to: {output_file}")
    print(f"{'='*60}\n")
    print("=== Final Code ===")
    print(code)


def main():
    print("=== Multi-Agent Code Generator (GLM-4.7) ===\n")
    requirement = input("Enter your requirement:\n> ").strip()
    if not requirement:
        print("Requirement cannot be empty.")
        return

    output_file = input("\nOutput filename (e.g. output.py): ").strip()
    if not output_file:
        output_file = "output.py"

    run_agent_loop(requirement, output_file)


if __name__ == "__main__":
    main()
