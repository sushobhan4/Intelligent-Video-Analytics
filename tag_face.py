"""
================================================================================
Neural CCTV Analyser - Face Enrollment & Tagging Utility
================================================================================
CLI tool to manage local ChromaDB face vectors:
  - List enrolled household members
  - View captured unknown face snapshots
  - Tag an unknown face snapshot as a family member (e.g. "Father", "Mother")
  - Directly enroll a new face from any photo file
"""

import os
import sys
import argparse
from neural_cctv.face_engine import FaceVectorEngine


def main():
    parser = argparse.ArgumentParser(description="Neural CCTV Analyser - Face Enrollment & Tagging Utility")
    parser.add_argument("--list", action="store_true", help="List all enrolled household identities")
    parser.add_argument("--unknowns", action="store_true", help="List pending unidentified face snapshots")
    parser.add_argument("--tag", type=str, help="Path or filename of an unknown face snapshot to name")
    parser.add_argument("--name", type=str, help="Name to assign (e.g. 'Father', 'Mother', 'Uncle Joe')")
    parser.add_argument("--enroll", type=str, help="Path to any photo file to enroll directly")
    parser.add_argument("--relation", type=str, default="Family", help="Relation/role (default: 'Family')")
    parser.add_argument("--db-path", type=str, default="./data/face_db", help="Path to local ChromaDB store")
    parser.add_argument("--unknown-dir", type=str, default="./data/unknown_faces", help="Path to unknown snapshots")

    args = parser.parse_args()

    engine = FaceVectorEngine(
        db_path=args.db_path,
        unknown_dir=args.unknown_dir
    )

    # 1. List known enrolled faces
    if args.list:
        faces = engine.list_known_faces()
        print("\n" + "=" * 65)
        print("         NEURAL CCTV ANALYSER - ENROLLED IDENTITIES")
        print("=" * 65)
        if not faces:
            print(" No faces currently registered in local ChromaDB database.")
            print(" Enroll a face with: py tag_face.py --enroll <photo.jpg> --name 'Father'")
        else:
            for f in faces:
                print(f" • Name: {f['name']:<18} | Role: {f['relation']:<12} | ID: {f['id']}")
        print("=" * 65 + "\n")
        return

    # 2. List pending unknown faces
    if args.unknowns:
        unknowns = engine.list_unidentified_snapshots()
        print("\n" + "=" * 75)
        print("         NEURAL CCTV ANALYSER - UNIDENTIFIED FACE SNAPSHOTS")
        print("=" * 75)
        if not unknowns:
            print(" No unidentified faces currently pending.")
        else:
            for idx, path in enumerate(unknowns, 1):
                fname = os.path.basename(path)
                print(f" [{idx}] {fname}")
                print(f"     Path: {path}")
            print("\n To name any face, run:")
            print(" py tag_face.py --tag <snapshot_filename> --name 'Father'")
        print("=" * 75 + "\n")
        return

    # 3. Tag an unknown snapshot
    if args.tag:
        if not args.name:
            print("[ERROR] Please provide a name using --name 'PersonName'")
            sys.exit(1)

        target_path = args.tag
        if not os.path.exists(target_path):
            candidate = os.path.join(args.unknown_dir, args.tag)
            if os.path.exists(candidate):
                target_path = candidate
            else:
                print(f"[ERROR] Snapshot file not found: {args.tag}")
                sys.exit(1)

        print(f"[Neural Face] Processing face snapshot: {target_path} ...")
        success = engine.enroll_face(name=args.name, image_bgr_or_path=target_path, relation=args.relation)
        if success:
            print(f"[SUCCESS] Successfully enrolled '{args.name}' into local ChromaDB!")
            print(f"[Neural Face] Future appearances will be identified as: '{args.name}'.")
        else:
            print("[ERROR] Could not detect a clear face in the specified image.")
        return

    # 4. Enroll direct photo
    if args.enroll:
        if not args.name:
            print("[ERROR] Please provide a name using --name 'PersonName'")
            sys.exit(1)

        if not os.path.exists(args.enroll):
            print(f"[ERROR] Image file not found: {args.enroll}")
            sys.exit(1)

        print(f"[Neural Face] Enrolling photo: {args.enroll} for '{args.name}' ...")
        success = engine.enroll_face(name=args.name, image_bgr_or_path=args.enroll, relation=args.relation)
        if success:
            print(f"[SUCCESS] Face for '{args.name}' enrolled into local ChromaDB!")
        else:
            print("[ERROR] Face embedding could not be generated. Please ensure face is visible.")
        return

    # If no flags provided, show help
    parser.print_help()


if __name__ == "__main__":
    main()
