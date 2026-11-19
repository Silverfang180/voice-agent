"""Comprehensive test script for Groq LLM AI Receptionist Brain."""
import argparse
import sys
import time

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from groq_llm import ReceptionistBrain


HINDI_TEST_QUESTIONS = [
    "आपका कार्यालय कितने बजे खुलता है?",
    "क्या आप रविवार को खुले रहते हैं?",
    "आपकी कौन-कौन सी सेवाएं हैं?",
    "कंप्यूटर रिपेयर की फीस कितनी है?",
]

MARATHI_TEST_QUESTIONS = [
    "तुमचं कार्यालय किती वाजता उघडतं?",
    "रविवारी कार्यालय सुरू असतं का?",
    "तुम्ही कोणत्या सेवा देता?",
    "संगणक दुरुस्तीचं शुल्क किती आहे?",
]

OUT_OF_FAQ_QUESTIONS = [
    ("hi", "क्या आपके पास बेसमेंट में कार पार्किंग की सुविधा है?"),
    ("mr", "तुमच्या ऑफिसमध्ये कॅफेटेरिया किंवा जेवणाची सोय उपलब्ध आहे का?"),
]

FOLLOW_UP_CONVERSATIONS = [
    {
        "language": "hi",
        "session_id": "session_hi_memory_test",
        "turns": [
            "कंप्यूटर रिपेयर की फीस कितनी है?",
            "और लैपटॉप रिपेयर की कितनी है?",
            "क्या आप इस काम के लिए घर पर भी आते हैं?",
        ],
    },
    {
        "language": "mr",
        "session_id": "session_mr_memory_test",
        "turns": [
            "तुमचं कार्यालय कुठे आहे?",
            "तिथे संपर्क करण्यासाठी फोन नंबर काय आहे?",
        ],
    },
]


def run_benchmark():
    print("=" * 65)
    print("AI RECEPTIONIST BRAIN BENCHMARK (GROQ LLM)")
    print("=" * 65)

    brain = ReceptionistBrain()
    print(f"Model Selected:     {brain.model_name}")
    print(f"Model Verified:     {brain.verify_model_available()}")
    print("=" * 65 + "\n")

    results = []

    # 1. Hindi Standard Questions
    print("--- 1. HINDI STANDARD FAQ TESTS ---")
    for q in HINDI_TEST_QUESTIONS:
        resp = brain.ask(q, session_id="hindi_std_session", language="hi")
        results.append(resp)
        print(f"Q: {q}")
        print(f"A: {resp.get('reply')}")
        print(f"Response Time: {resp.get('response_time', 0):.2f} s\n")

    # 2. Marathi Standard Questions
    print("--- 2. MARATHI STANDARD FAQ TESTS ---")
    for q in MARATHI_TEST_QUESTIONS:
        resp = brain.ask(q, session_id="marathi_std_session", language="mr")
        results.append(resp)
        print(f"Q: {q}")
        print(f"A: {resp.get('reply')}")
        print(f"Response Time: {resp.get('response_time', 0):.2f} s\n")

    # 3. Out of FAQ Questions (Hallucination Prevention)
    print("--- 3. OUT-OF-FAQ HALLUCINATION PREVENTION TESTS ---")
    for lang, q in OUT_OF_FAQ_QUESTIONS:
        resp = brain.ask(q, session_id=f"oof_{lang}", language=lang)
        results.append(resp)
        print(f"[{lang.upper()}] Q (Unlisted): {q}")
        print(f"A: {resp.get('reply')}")
        print(f"Response Time: {resp.get('response_time', 0):.2f} s\n")

    # 4. Multi-Turn Conversation Memory Test
    print("--- 4. CONVERSATION CONTEXT & FOLLOW-UP MEMORY TESTS ---")
    for conv in FOLLOW_UP_CONVERSATIONS:
        lang = conv["language"]
        sess_id = conv["session_id"]
        print(f"\n[Session: {sess_id} ({lang.upper()})]")
        for idx, turn in enumerate(conv["turns"], 1):
            resp = brain.ask(turn, session_id=sess_id, language=lang)
            results.append(resp)
            print(f"Turn {idx} Q: {turn}")
            print(f"Turn {idx} A: {resp.get('reply')}")
            print(f"Response Time: {resp.get('response_time', 0):.2f} s (Turn {resp.get('turn_count')})\n")

    # Benchmark Summary
    valid_times = [r["response_time"] for r in results if "response_time" in r]
    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
    min_time = min(valid_times) if valid_times else 0
    max_time = max(valid_times) if valid_times else 0

    print("=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Total Queries Evaluated: {len(results)}")
    print(f"Average Response Time:   {avg_time:.2f} s")
    print(f"Fastest Response Time:   {min_time:.2f} s")
    print(f"Slowest Response Time:   {max_time:.2f} s")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Test Groq LLM AI Receptionist.")
    parser.add_argument("--query", type=str, default=None, help="Custom single question to ask.")
    parser.add_argument("--language", type=str, choices=["hi", "mr"], default="hi", help="Language code.")
    args = parser.parse_args()

    if args.query:
        brain = ReceptionistBrain()
        resp = brain.ask(args.query, session_id="custom_cli", language=args.language)
        print(f"Language:      {args.language}")
        print(f"Question:      {args.query}")
        print(f"Answer:        {resp.get('reply')}")
        print(f"Response Time: {resp.get('response_time', 0):.2f} s")
    else:
        run_benchmark()


if __name__ == "__main__":
    main()

