#!/usr/bin/env python3
"""Integration test: CFQ generator with reasoning_content on GLM-4.6 traj."""

import json
import re
import sys
from pathlib import Path

import requests

from minisweagent.utils.cfq_generator import CFQGenerator, CFQGeneratorConfig

TRAJ = Path(__file__).resolve().parents[1] / "runs/with-pruner-GLM-4.6/astropy__astropy-7606/astropy__astropy-7606.traj.json"
PRUNER_URL = "http://10.10.10.39:6001/prune"
PRUNER_MIN_OUTPUT_CHARS = 1600
CFQ_MIN_OUTPUT_CHARS = 3200


def extract_reasoning(msg: dict) -> str:
    try:
        message = msg["extra"]["response"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    reasoning = message.get("reasoning_content") or ""
    if not reasoning:
        reasoning = (message.get("provider_specific_fields") or {}).get("reasoning_content") or ""
    return reasoning.strip()


def parse_command(content: str) -> str:
    match = re.search(r"```bash\n(.*?)\n```", content, re.S)
    return match.group(1).strip() if match else ""


def extract_thought(content: str) -> str:
    match = re.match(r"THOUGHT:\s*(.*?)(?=```bash)", content, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def get_command_output(traj: dict, assistant_index: int) -> str:
    ai = 0
    for i, msg in enumerate(traj["messages"]):
        if msg["role"] != "assistant":
            continue
        if ai == assistant_index:
            for follow_up in traj["messages"][i + 1 :]:
                if follow_up["role"] == "user":
                    match = re.search(r"<output>\n?(.*?)\n?</output>", follow_up["content"], re.S)
                    return match.group(1) if match else ""
            return ""
        ai += 1
    return ""


def main() -> int:
    traj = json.loads(TRAJ.read_text())
    gen = CFQGenerator(CFQGeneratorConfig())

    steps = []
    assistant_idx = 0
    for msg in traj["messages"]:
        if msg["role"] != "assistant":
            continue
        command = parse_command(msg["content"])
        if not command:
            continue
        output = get_command_output(traj, assistant_idx)
        assistant_idx += 1
        reasoning = extract_reasoning(msg)
        thought = extract_thought(msg["content"])
        should = gen.should_generate(command, output, min_output_chars=CFQ_MIN_OUTPUT_CHARS)
        steps.append(
            {
                "step": assistant_idx,
                "command": command,
                "output_len": len(output),
                "reasoning": reasoning,
                "thought": thought,
                "should_generate": should,
                "will_generate_cfq": should and bool(reasoning),
                "output": output,
            }
        )

    print("=" * 72)
    print("GLM-4.6 astropy-7606: auto-CFQ trigger analysis")
    print("=" * 72)
    print(f"{'Step':>4} {'CFQ?':>6} {'RC?':>4} {'OutLen':>7}  Command")
    print("-" * 72)
    for step in steps:
        cfq_flag = "YES" if step["will_generate_cfq"] else ("skip" if not step["should_generate"] else "noRC")
        rc = "yes" if step["reasoning"] else "no"
        print(f"{step['step']:4d} {cfq_flag:>6} {rc:>4} {step['output_len']:7d}  {step['command'][:55]}")

    print("\n" + "=" * 72)
    print("Live CFQ generation")
    print("=" * 72)

    generated = []
    prior_reasonings: list[str] = []
    for step in steps:
        step["prior_reasoning"] = prior_reasonings.copy()
        if step["reasoning"]:
            prior_reasonings.append(step["reasoning"])
        if not step["will_generate_cfq"]:
            continue
        block, used_prior = gen.build_reasoning_block(
            step["reasoning"],
            step["prior_reasoning"],
            min_chars=gen.config.min_reasoning_chars,
        )
        cfq = gen.generate(
            reasoning=step["reasoning"],
            command=step["command"],
            prior_reasoning=step["prior_reasoning"],
        )
        step["cfq"] = cfq
        step["used_prior_context"] = used_prior
        generated.append(step)
        print(f"\n[Step {step['step']}] {step['command']}")
        print(f"  reasoning_content: {step['reasoning'][:200]}")
        if used_prior:
            print(f"  prior context ({len(step['prior_reasoning'])} steps):")
            for prior in step["prior_reasoning"][-3:]:
                print(f"    - {prior[:120]}")
        print(f"  CFQ: {cfq}")

    print("\n" + "=" * 72)
    print(f"Summary: {len(generated)}/{len(steps)} steps get auto-CFQ")
    print("=" * 72)

    print("\n" + "=" * 72)
    print("Pruner end-to-end (first 2 generated CFQs)")
    print("=" * 72)
    for step in generated[:2]:
        if not step.get("cfq"):
            continue
        try:
            response = requests.post(
                PRUNER_URL,
                json={
                    "query": step["cfq"],
                    "code": step["output"],
                    "threshold": 0.5,
                    "chunk_overlap_tokens": 50,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            print(
                f"\n[Step {step['step']}] {data.get('origin_token_cnt')} -> {data.get('left_token_cnt')} tokens "
                f"(score={data.get('score', 0):.3f})"
            )
            print(f"  CFQ: {step['cfq']}")
            print(f"  Pruned preview:\n{(data.get('pruned_code') or '')[:400]}")
        except Exception as exc:
            print(f"\n[Step {step['step']}] Pruner unavailable: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
