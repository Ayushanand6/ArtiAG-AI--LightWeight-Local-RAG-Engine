#!/usr/bin/env python3
"""
ArtiAG Evaluation Script
Measures:
1. Retrieval Hit Rate (Recall): Expected doc prefix in top-K matches (or 0 matches for out-of-scope)
2. Answer Accuracy: Expected keyword present in generated response
"""

import json
import os
import sys
import time
import requests

BASE_URL = "http://localhost:8080"
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data.json")

def load_data():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def wait_for_server():
    print(f"Checking server availability at {BASE_URL}...")
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/api/status")
            if r.status_code == 200:
                print("Server is up and reachable.")
                return True
        except Exception:
            try:
                r = requests.get(f"{BASE_URL}/api/documents/ingest")
                if r.status_code in [200, 400, 405, 500]:
                    print("Server is up and reachable.")
                    return True
            except Exception:
                pass
        time.sleep(1)
    print("Error: ArtiAG server is not running at", BASE_URL)
    return False

def ingest_documents(documents):
    print(f"\n--- Ingesting {len(documents)} Documents ---")
    start = time.time()
    total_chunks = 0
    for doc in documents:
        res = requests.post(
            f"{BASE_URL}/api/documents/ingest",
            json={"id": doc["id"], "text": doc["text"]}
        )
        if res.status_code == 200:
            data = res.json()
            chunks = data.get("chunksIndexed", 1)
            total_chunks += chunks
            print(f"  [OK] Ingested '{doc['id']}' ({chunks} chunk(s))")
        else:
            print(f"  [FAIL] Failed to ingest '{doc['id']}': {res.text}")
    elapsed = time.time() - start
    print(f"Finished ingesting {len(documents)} docs ({total_chunks} total chunks) in {elapsed:.2f}s\n")

def run_evaluation(questions):
    print("--- Running Evaluation on Queries ---")
    retrieval_hits = 0
    answer_hits = 0
    total = len(questions)

    results = []

    for i, item in enumerate(questions, 1):
        q = item["question"]
        expected_id = item["expected_doc_id"]
        expected_kw = item["expected_keyword"].lower()

        t0 = time.time()
        res = requests.post(
            f"{BASE_URL}/api/chat/query",
            json={"query": q, "topK": 3}
        )
        latency_ms = (time.time() - t0) * 1000

        if res.status_code != 200:
            print(f"[{i:02d}/{total}] ERROR {res.status_code} for query: {q}")
            results.append({"q": q, "retrieval": False, "answer": False, "latency": latency_ms})
            continue

        data = res.json()
        answer = data.get("answer", "")
        matches = data.get("matches", [])

        # 1. Retrieval Hit Logic:
        # If expected_id is None (out of scope), retrieval is a hit if matches is empty (guardrail worked)
        if expected_id is None:
            retrieval_ok = (len(matches) == 0)
        else:
            prefix = expected_id + "#"
            retrieval_ok = any(m["record"]["id"].startswith(prefix) for m in matches)

        # 2. Answer Accuracy Logic:
        answer_ok = (expected_kw in answer.lower())

        if retrieval_ok:
            retrieval_hits += 1
        if answer_ok:
            answer_hits += 1

        status_r = "PASS" if retrieval_ok else "FAIL"
        status_a = "PASS" if answer_ok else "FAIL"

        print(f"[{i:02d}/{total}] Retrieval: {status_r} | Answer: {status_a} | Latency: {latency_ms:.0f}ms")
        print(f"       Q: {q}")
        print(f"       Expected: id={expected_id}, keyword='{expected_kw}'")
        print(f"       Got: {answer.strip()[:100]}...\n")

        results.append({
            "q": q,
            "retrieval": retrieval_ok,
            "answer": answer_ok,
            "latency": latency_ms
        })

    retrieval_rate = (retrieval_hits / total) * 100
    answer_rate = (answer_hits / total) * 100
    avg_latency = sum(r["latency"] for r in results) / total

    print("=" * 60)
    print("                    EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Test Questions     : {total}")
    print(f"Retrieval Hit Rate (Recall) : {retrieval_hits}/{total} ({retrieval_rate:.1f}%)")
    print(f"Answer Correctness Rate    : {answer_hits}/{total} ({answer_rate:.1f}%)")
    print(f"Average Query Latency      : {avg_latency:.1f} ms")
    print("=" * 60)

    return {
        "total": total,
        "retrieval_hits": retrieval_hits,
        "retrieval_rate": retrieval_rate,
        "answer_hits": answer_hits,
        "answer_rate": answer_rate,
        "avg_latency_ms": avg_latency
    }

if __name__ == "__main__":
    data = load_data()
    if not wait_for_server():
        sys.exit(1)
    ingest_documents(data["documents"])
    run_evaluation(data["questions"])
