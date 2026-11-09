import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from clip_vector_db import CCTVVectorStore

def run_test():
    store = CCTVVectorStore()

    test_report = """TIMESTAMP: 2026-11-04 14:22:15
TITLE: Adult male in grey hoodie walking toward front gate
OVERALL DIRECTION OF MOTION: Heading south along driveway towards gate
CONSOLIDATED FORENSIC ENTITY BREAKDOWN:
- Person: Adult male, medium build, dark grey hooded sweatshirt with white drawstrings, dark blue denim jeans, black sneakers. Carrying a black backpack over right shoulder.
FINAL ACTION SUMMARY:
Subject entered the driveway at 14:22:15, glanced briefly toward the camera, proceeded along the driveway, and exited through the front gate."""

    print("Adding test clip...")
    store.add_clip(
        clip_id="event_clip_05",
        forensic_report=test_report,
        metadata={
            "camera_id": "FRONT_PORCH_CAM_01",
            "clip_path": "test/clips/event_clip_05_04m50s_to_04m59s.mp4",
            "duration_sec": 9.0,
            "start_sec": 290.0,
            "end_sec": 299.0
        }
    )
    print(f"Total clips stored: {store.count()}")

    query = "man carrying backpack wearing grey hoodie"
    print(f"Searching for: '{query}'")
    results = store.search_clips(query, n_results=1)
    if results:
        r = results[0]
        print(f"[+] Match found! ID: {r['clip_id']}, Similarity: {r['similarity']}")
        print(f"    Title  : {r['metadata'].get('title')}")
        print(f"    Action : {r['metadata'].get('action_summary')}")

    print("Resetting database back to clean empty state...")
    store.reset_db()
    print(f"[+] Final clean clip count in ChromaDB: {store.count()}")

if __name__ == "__main__":
    run_test()
