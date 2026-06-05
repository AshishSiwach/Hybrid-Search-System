"""
find_queries.py

Finds good demo queries from your MS MARCO dataset — both climate/energy
queries and general non-climate queries — so you can pick the best one
to use as the default in demo.py.

Run from your project root:
    python find_queries.py
"""

import json
from collections import defaultdict

with open("data/ms_marco_queries.json", encoding = "utf-8") as f:
    queries = json.load(f)

print(f"Total queries in dataset: {len(queries)}")
print()

# ------------------------------------------------------------------
# Climate queries
# ------------------------------------------------------------------
climate_qs = [
    q for q in queries
    if q.get("is_climate") and sum(q["relevance"].values()) > 0
]

print("=" * 70)
print("CLIMATE / ENERGY QUERIES")
print(f"Found {len(climate_qs)} climate queries with at least 1 relevant passage")
print("=" * 70)
for q in climate_qs[:15]:
    n_rel = sum(q["relevance"].values())
    print(f"  [{n_rel} relevant]  {q['query']}")

# ------------------------------------------------------------------
# Non-climate queries — sample from diverse topics
# ------------------------------------------------------------------
non_climate_qs = [
    q for q in queries
    if not q.get("is_climate") and sum(q["relevance"].values()) > 0
]

# Group by rough topic using first keyword
topic_buckets = defaultdict(list)
topic_keywords = {
    "health":      ["disease", "cancer", "symptoms", "treatment", "medical", "health", "drug", "pain", "diabetes", "vitamin"],
    "science":     ["physics", "chemistry", "biology", "space", "planet", "gravity", "atom", "dna", "evolution", "quantum"],
    "history":     ["war", "history", "ancient", "president", "century", "empire", "revolution", "battle", "civilization"],
    "technology":  ["computer", "software", "internet", "ai", "algorithm", "programming", "machine", "robot", "data"],
    "geography":   ["country", "capital", "continent", "population", "city", "mountain", "river", "ocean", "island"],
    "food":        ["food", "recipe", "cook", "nutrition", "diet", "calories", "protein", "vitamin", "eat"],
    "sports":      ["sport", "football", "basketball", "soccer", "tennis", "olympic", "player", "game", "team"],
    "finance":     ["money", "economy", "bank", "tax", "invest", "stock", "inflation", "gdp", "budget"],
    "general":     [],
}

for q in non_climate_qs:
    assigned = False
    for topic, keywords in topic_keywords.items():
        if topic == "general":
            continue
        if any(kw in q["query"].lower() for kw in keywords):
            topic_buckets[topic].append(q)
            assigned = True
            break
    if not assigned:
        topic_buckets["general"].append(q)

print()
print("=" * 70)
print("NON-CLIMATE QUERIES  (sample across topics)")
print(f"Found {len(non_climate_qs)} non-climate queries with relevant passages")
print("=" * 70)

for topic, qs in sorted(topic_buckets.items()):
    if not qs:
        continue
    # Pick the 2 shortest (most concise) queries per topic
    sample = sorted(qs, key=lambda q: len(q["query"]))[:2]
    print(f"\n  -- {topic.upper()} ({len(qs)} queries) --")
    for q in sample:
        n_rel = sum(q["relevance"].values())
        print(f"  [{n_rel} relevant]  {q['query']}")

# ------------------------------------------------------------------
# Top picks summary
# ------------------------------------------------------------------
print()
print("=" * 70)
print("TOP PICKS FOR DEMO  (copy one of these as --query)")
print("=" * 70)

# Best climate query — most relevant passages
if climate_qs:
    best_climate = sorted(climate_qs, key=lambda q: sum(q["relevance"].values()), reverse=True)[0]
    print(f"\n  Best climate query ({sum(best_climate['relevance'].values())} relevant passages):")
    print(f"  python demo.py --query \"{best_climate['query']}\"")

# Best general query — most relevant passages, short text
best_general = sorted(non_climate_qs, key=lambda q: (sum(q["relevance"].values()), -len(q["query"])), reverse=True)[0]
print(f"\n  Best general query ({sum(best_general['relevance'].values())} relevant passages):")
print(f"  python demo.py --query \"{best_general['query']}\"")

# 3 diverse general picks
print("\n  3 diverse general picks:")
seen_topics = set()
picks = []
for q in sorted(non_climate_qs, key=lambda q: sum(q["relevance"].values()), reverse=True):
    topic = next((t for t, ks in topic_keywords.items() if t != "general" and any(k in q["query"].lower() for k in ks)), "general")
    if topic not in seen_topics:
        picks.append((topic, q))
        seen_topics.add(topic)
    if len(picks) == 3:
        break

for topic, q in picks:
    print(f"  [{topic}]  python demo.py --query \"{q['query']}\"")

print()