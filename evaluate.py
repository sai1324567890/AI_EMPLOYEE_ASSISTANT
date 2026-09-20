import argparse
import csv
from pathlib import Path

from src.assistant import TrainingAssistant
from src import config


def gold_source_filename(gold_source: str) -> str:
    if not gold_source:
        return ""
    return Path(gold_source).name


def run_evaluation(retrieval_backend: str = None, router_backend: str = None) -> dict:
    """Runs the full evaluation set through the assistant and returns a
    dict with the summary metrics, per-question rows, and the assistant's
    backend info. Used by both the CLI script below and
    generate_report.py. Each question gets its own session_id so that
    unrelated evaluation questions never leak into each other's
    conversation-memory context, and analytics logging is disabled so
    evaluation runs don't pollute the Admin Dashboard's real usage data.
    """
    assistant = TrainingAssistant(
        retrieval_backend=retrieval_backend,
        router_backend=router_backend,
        log_analytics=False,
    )

    rows = []
    with open(config.EVAL_SET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results = []
    correct_routes = 0
    citation_checks = 0
    citation_hits = 0
    keyphrase_checks = 0
    keyphrase_hits = 0

    for row in rows:
        qid = row["question_id"]
        question = row["question"]
        expected_route = row["expected_route"]
        gold_source = row.get("gold_source", "")
        key_phrase = row.get("gold_answer_key_phrase_or_expected_behavior", "")

        # unique session per question -> no cross-question memory bleed
        result = assistant.ask(question, session_id=f"eval-{qid}")

        route_ok = result.route == expected_route
        correct_routes += int(route_ok)

        citation_ok = None
        gold_fname = gold_source_filename(gold_source)
        if gold_fname:
            citation_checks += 1
            hit = any(gold_fname in c for c in result.citations)
            citation_hits += int(hit)
            citation_ok = hit

        keyphrase_ok = None
        if key_phrase and expected_route != "direct_llm":
            keyphrase_checks += 1
            phrases = [p.strip().lower() for p in key_phrase.split(";") if p.strip()]
            hit = any(p in result.answer.lower() for p in phrases) if phrases else False
            keyphrase_hits += int(hit)
            keyphrase_ok = hit

        results.append({
            "question_id": qid,
            "question": question,
            "expected_route": expected_route,
            "predicted_route": result.route,
            "route_correct": route_ok,
            "routing_confidence": round(result.routing_confidence, 3),
            "gold_source": gold_source,
            "predicted_citations": "; ".join(result.citations),
            "citation_correct": citation_ok,
            "key_phrase": key_phrase,
            "key_phrase_found": keyphrase_ok,
            "answer": result.answer.replace("\n", " ")[:400],
            "answer_length_chars": len(result.answer),
        })

    n = len(rows)
    summary = {
        "n": n,
        "backend": assistant.llm.backend,
        "retrieval_backend": assistant.retrieval_backend,
        "router_backend": assistant.router_backend,
        "routing_accuracy": correct_routes / n if n else 0.0,
        "correct_routes": correct_routes,
        "citation_checks": citation_checks,
        "citation_hits": citation_hits,
        "citation_match_rate": (citation_hits / citation_checks) if citation_checks else None,
        "keyphrase_checks": keyphrase_checks,
        "keyphrase_hits": keyphrase_hits,
        "keyphrase_presence_rate": (keyphrase_hits / keyphrase_checks) if keyphrase_checks else None,
    }
    return {"summary": summary, "rows": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the AI Training Assistant.")
    parser.add_argument("--retrieval", choices=["tfidf", "faiss"], default=None,
                         help="Override the retrieval backend (default: from config/.env).")
    parser.add_argument("--router", choices=["rule_based", "llm"], default=None,
                         help="Override the router backend (default: from config/.env).")
    args = parser.parse_args()

    outcome = run_evaluation(retrieval_backend=args.retrieval, router_backend=args.router)
    summary = outcome["summary"]
    results = outcome["rows"]

    out_path = config.REPORTS_DIR / "eval_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n = summary["n"]
    print(f"LLM backend: {summary['backend']}")
    print(f"Retrieval backend: {summary['retrieval_backend']}")
    print(f"Router backend: {summary['router_backend']}")
    print(f"Total questions evaluated: {n}")
    print(f"Routing accuracy: {summary['correct_routes']}/{n} = {summary['routing_accuracy']:.1%}")
    if summary["citation_checks"]:
        print(f"Citation match rate (RAG routes): {summary['citation_hits']}/{summary['citation_checks']} "
              f"= {summary['citation_match_rate']:.1%}")
    if summary["keyphrase_checks"]:
        print(f"Key-phrase presence rate: {summary['keyphrase_hits']}/{summary['keyphrase_checks']} "
              f"= {summary['keyphrase_presence_rate']:.1%}")
    print(f"\nDetailed results written to: {out_path}")


if __name__ == "__main__":
    main()
