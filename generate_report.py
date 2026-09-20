"""
Generates the full Evaluation & Testing Report required by the capstone
spec: Accuracy Metrics, Response Quality Analysis, Test Cases, and User
Feedback Results. Writes reports/EVALUATION_REPORT.md (and reuses
evaluate.py's run to also refresh reports/eval_results.csv).

Usage:
    python generate_report.py
    python generate_report.py --retrieval faiss --router llm
"""
import argparse
from datetime import datetime, timezone
from statistics import mean, pstdev

from evaluate import run_evaluation
from src import config


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def _response_quality_section(rows) -> str:
    lengths = [r["answer_length_chars"] for r in rows]
    by_route = {}
    for r in rows:
        by_route.setdefault(r["predicted_route"], []).append(r)

    lines = [
        "## 2. Response Quality Analysis\n",
        f"- **Average answer length:** {mean(lengths):.0f} characters "
        f"(σ = {pstdev(lengths):.0f})",
        f"- **Shortest / longest answer:** {min(lengths)} / {max(lengths)} characters",
    ]

    citations_present = sum(1 for r in rows if r["predicted_citations"])
    lines.append(
        f"- **Answers with at least one citation:** {citations_present}/{len(rows)} "
        f"({citations_present / len(rows):.1%})"
    )

    lines.append("\n**Breakdown by predicted route:**\n")
    lines.append("| Route | Questions | Avg. answer length | With citation |")
    lines.append("|---|---|---|---|")
    for route, items in sorted(by_route.items()):
        avg_len = mean(i["answer_length_chars"] for i in items)
        with_cite = sum(1 for i in items if i["predicted_citations"])
        lines.append(f"| `{route}` | {len(items)} | {avg_len:.0f} chars | {with_cite}/{len(items)} |")

    return "\n".join(lines) + "\n"


def _test_cases_section(rows) -> str:
    lines = [
        "## 3. Test Cases\n",
        "All 20 labeled questions from `data/evaluation_set.csv`, run end-to-end "
        "through routing -> retrieval -> generation.\n",
        "| ID | Question | Expected Route | Predicted Route | Route OK | Citation OK | Key-Phrase OK |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        route_ok = "✅" if r["route_correct"] else "❌"
        cite_ok = "✅" if r["citation_correct"] else ("❌" if r["citation_correct"] is False else "—")
        kp_ok = "✅" if r["key_phrase_found"] else ("❌" if r["key_phrase_found"] is False else "—")
        q_short = r["question"] if len(r["question"]) <= 70 else r["question"][:67] + "..."
        lines.append(
            f"| {r['question_id']} | {q_short} | `{r['expected_route']}` | "
            f"`{r['predicted_route']}` | {route_ok} | {cite_ok} | {kp_ok} |"
        )
    return "\n".join(lines) + "\n"


def _feedback_section() -> str:
    lines = ["## 4. User Feedback Results\n"]
    try:
        from src import analytics
        summary = analytics.get_feedback_summary()
    except Exception:
        summary = None

    if not summary:
        lines.append(
            "No user feedback has been recorded yet. Feedback accumulates automatically "
            "as real users click 👍 / 👎 on answers in the Streamlit app (`app.py`) - "
            "re-run this report after some usage to populate this section."
        )
        return "\n".join(lines) + "\n"

    lines.append(f"- **Total rated answers:** {summary['total']}")
    lines.append(f"- **👍 Positive:** {summary['up']}  |  **👎 Negative:** {summary['down']}")
    lines.append(f"- **Satisfaction rate:** {summary['satisfaction_rate']:.1%}")
    if summary["comments"]:
        lines.append("\n**Recent comments:**\n")
        lines.append("| Question | Rating | Comment |")
        lines.append("|---|---|---|")
        for question, rating, comment, ts in summary["comments"]:
            emoji = "👍" if rating == "up" else "👎"
            q_short = question if len(question) <= 60 else question[:57] + "..."
            lines.append(f"| {q_short} | {emoji} | {comment} |")
    return "\n".join(lines) + "\n"


def build_report(retrieval_backend=None, router_backend=None) -> str:
    outcome = run_evaluation(retrieval_backend=retrieval_backend, router_backend=router_backend)
    summary = outcome["summary"]
    rows = outcome["rows"]

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = f"""# Evaluation & Testing Report
**AI Training Assistant for New Employees**

Generated: {generated_at}
LLM backend: `{summary['backend']}` · Retrieval backend: `{summary['retrieval_backend']}` · Router backend: `{summary['router_backend']}`

---

## 1. Accuracy Metrics

| Metric | Score |
|---|---|
| Routing accuracy | {summary['correct_routes']}/{summary['n']} = {_pct(summary['routing_accuracy'])} |
| Citation match rate (RAG routes) | {summary['citation_hits']}/{summary['citation_checks']} = {_pct(summary['citation_match_rate'])} |
| Key-phrase presence rate | {summary['keyphrase_hits']}/{summary['keyphrase_checks']} = {_pct(summary['keyphrase_presence_rate'])} |

Routing accuracy measures whether the query router selected the same route a human
labeler expected. Citation match rate measures whether the correct source document was
among the retrieved/cited chunks for document-grounded routes. Key-phrase presence rate
is a loose proxy for answer correctness/completeness (does the generated answer contain
at least one of the labeled key phrases).

"""

    body = "\n".join([
        _response_quality_section(rows),
        _test_cases_section(rows),
        _feedback_section(),
    ])

    footer = (
        "\n---\n\n## 5. How to reproduce\n\n"
        "```bash\n"
        "python generate_report.py                       # default backends (tfidf / rule_based)\n"
        "python generate_report.py --retrieval faiss      # FAISS vector-DB retrieval\n"
        "python generate_report.py --router llm           # LLM-based query classification\n"
        "```\n"
    )

    return header + body + footer


def main():
    parser = argparse.ArgumentParser(description="Generate the Evaluation & Testing Report.")
    parser.add_argument("--retrieval", choices=["tfidf", "faiss"], default=None)
    parser.add_argument("--router", choices=["rule_based", "llm"], default=None)
    args = parser.parse_args()

    report_text = build_report(retrieval_backend=args.retrieval, router_backend=args.router)

    out_path = config.REPORTS_DIR / "EVALUATION_REPORT.md"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Evaluation & Testing Report written to: {out_path}")


if __name__ == "__main__":
    main()
