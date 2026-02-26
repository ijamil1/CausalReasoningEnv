"""Filter eval results where format_compliance != 1 or completion is empty."""

import json
import sys

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "environments/CausalReasoningEnv_1/outputs/evals/"
    "CausalReasoningEnv_1--Qwen--Qwen3-VL-8B-Instruct/86576691/results.jsonl"
)

failures = []
passing_rewards = []
total = 0

with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        total += 1
        if obj.get("format_compliance") != 1.0 or len(obj.get("completion", [])) == 0:
            failures.append(obj)
        else:
            passing_rewards.append(obj.get("reward", 0.0))

avg_reward = sum(passing_rewards) / len(passing_rewards) if passing_rewards else float("nan")

print(f"Total rollouts : {total}")
print(f"Failures       : {len(failures)}")
print(f"Passing        : {len(passing_rewards)}  (avg reward: {avg_reward:.4f})")
print()

for obj in failures:
    print(f"─── example_id={obj['example_id']} ───")
    print(f"  format_compliance : {obj.get('format_compliance')}")
    print(f"  completion length : {len(obj.get('completion', []))}")
    print(f"  reward            : {obj.get('reward')}")
    if obj.get("completion"):
        last_msg = obj["completion"][-1]
        content = last_msg.get("content", "")
        print(f"  last content      : {content[:300]!r}")
    print()
