"""Run the test-case table (guidance steps 9-11) and record results in results/evaluation.md.

Usage (from the project root, after training):
    python src/run_tests.py
"""
import os

from predict import predict_message

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EVAL_PATH = os.path.normpath(os.path.join(ROOT, "results", "evaluation.md"))
MARKER = "## Step 11: Test with new messages"

# (id, scenario, message, expected)  expected: 'spam' | 'ham' | 'no_prediction'
# 'tricky' cases are genuinely hard: failures there are informative, not bugs.
CASES = [
    ("TC_SPAM_01", "Classify an obvious spam message", "You won a free prize! Click now to claim.", "spam"),
    ("TC_SPAM_02", "Classify a normal personal message", "Are we meeting at 5 pm today?", "ham"),
    ("TC_SPAM_03", "Enter an empty message", "", "no_prediction"),
    ("TC_SPAM_04", "Whitespace-only message", "    ", "no_prediction"),
    ("TC_SPAM_05", "Only symbols / numbers", "!!! ??? 12345", "no_prediction"),
    ("TC_SPAM_06", "Phishing: blocked account", "Urgent! Your account is blocked. Verify your details immediately.", "spam"),
    ("TC_SPAM_07", "Normal request for notes", "Please send me the notes from yesterday's class.", "ham"),
    ("TC_SPAM_08", "Very short ham", "ok", "ham"),
    ("TC_SPAM_09", "Very short spam", "WINNER! Claim prize now", "spam"),
    ("TC_SPAM_10", "Hinglish ham", "Kal library pe milte hain, 5 baje.", "ham"),
    ("TC_SPAM_11", "Loan spam", "Get a loan of Rs 50000 approved in 5 minutes! Apply: bit.ly/3xFr9Q", "spam"),
    ("TC_SPAM_12", "Genuine promotional (order update)", "Your order has shipped and will arrive tomorrow. Track it in the app.", "ham"),
    ("TC_SPAM_13", "Genuine message with spam-like words", "I won the match today! Are you free for dinner after?", "ham"),
    ("TC_SPAM_14", "Disguised spam, no URL", "Hello dear, I have a gift for you. Reply to know more.", "spam"),
]


def main():
    rows, passed = [], 0
    for cid, scenario, msg, expected in CASES:
        r = predict_message(msg)
        got = "no_prediction" if not r["ok"] else r["label"]
        ok = got == expected
        passed += ok
        rows.append((cid, scenario, msg, expected, got, "Pass" if ok else "Fail"))
        print(f"{cid}: {'Pass' if ok else 'FAIL'}  expected={expected:13s} got={got}")

    lines = [MARKER, "",
             f"Result: **{passed}/{len(CASES)}** passed.", "",
             "| Test Case ID | Scenario | Message | Expected | Actual | Result |",
             "|---|---|---|---|---|---|"]
    for cid, scenario, msg, expected, got, res in rows:
        shown = msg.strip() or "(empty)"
        lines.append(f"| {cid} | {scenario} | `{shown}` | {expected} | {got} | {res} |")
    section = "\n".join(lines) + "\n"

    text = open(EVAL_PATH, encoding="utf-8").read() if os.path.exists(EVAL_PATH) else "# Evaluation Report\n"
    if MARKER in text:
        text = text.split(MARKER)[0].rstrip() + "\n\n"
    else:
        text = text.rstrip() + "\n\n"
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        f.write(text + section)
    print(f"\n{passed}/{len(CASES)} passed. Recorded in results/evaluation.md")


if __name__ == "__main__":
    main()
