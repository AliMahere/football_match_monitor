"""Turn reviewed verdicts into tracker quality numbers.

Reads the verdicts.csv produced by review_events.py once the verdict column is
filled, and reports how often identity actually survived the moments where it was
at risk. This is not HOTA: it is a targeted audit of the failure the brief asks
about, computed from a human decision on each event rather than from labels that
do not exist.

Example:
    python score_review.py --verdicts review_right/verdicts.csv --tracks final_right/tracks.json
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

GOOD = {"ok"}
BAD = {"swap", "split", "lost"}
SKIP = {"na", ""}


def parse(verdict):
    """Accept one verdict for the whole event, or one per identity.

    An event involves two tracks and they can fail independently, so all of these
    are valid:

        lost              the event failed, no further detail
        ok/lost           one identity held, the other was dropped
        2:ok,17:lost      spelled out per track id

    The event counts as survived only if every part is ok, because a crossing
    that loses one of two players is not a crossing the tracker handled.
    """
    text = verdict.strip().lower()
    if text in SKIP:
        return None
    parts = [p.strip() for p in text.replace("/", ",").replace("+", ",").split(",")]
    outcomes = {}
    for index, part in enumerate(parts):
        if not part:
            continue
        if ":" in part:
            track_id, value = part.split(":", 1)
            outcomes[track_id.strip()] = value.strip()
        else:
            outcomes[str(index)] = part
    if not outcomes or any(v not in GOOD | BAD for v in outcomes.values()):
        return None
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", required=True, type=Path)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--fps", type=float, default=45.0)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.verdicts.open()))
    judged, skipped = [], []
    for row in rows:
        outcomes = parse(row["verdict"])
        if outcomes is None:
            skipped.append(row)
        else:
            judged.append((row, outcomes))

    if not judged:
        raise SystemExit(
            f"no verdicts filled in {args.verdicts}\n"
            "put ok / swap / split / lost / na in the verdict column first\n"
            "for a mixed event use ok/lost, or 2:ok,17:lost"
        )

    by_type = defaultdict(lambda: {"ok": 0, "bad": 0, "detail": defaultdict(int)})
    identities_ok = identities_bad = 0
    for row, outcomes in judged:
        bucket = by_type[row["type"]]
        # The event survives only if every identity in it survived.
        survived = all(v in GOOD for v in outcomes.values())
        bucket["ok" if survived else "bad"] += 1
        for value in outcomes.values():
            bucket["detail"][value] += 1
            if value in GOOD:
                identities_ok += 1
            else:
                identities_bad += 1

    print(f"reviewed {len(judged)} of {len(rows)} events "
          f"({len(skipped)} left blank or marked na)\n")
    print(f"{'event type':<12} {'survived':>9} {'failed':>7} {'rate':>7}   breakdown")
    total_ok = total_bad = 0
    for name, bucket in sorted(by_type.items()):
        ok, bad = bucket["ok"], bucket["bad"]
        total_ok += ok
        total_bad += bad
        rate = ok / (ok + bad) if ok + bad else 0
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(bucket["detail"].items()))
        print(f"{name:<12} {ok:>9} {bad:>7} {rate:>6.1%}   {detail}")

    overall = total_ok / (total_ok + total_bad) if total_ok + total_bad else 0
    print(f"\n{'ALL':<12} {total_ok:>9} {total_bad:>7} {overall:>6.1%}")

    # An event with one identity held and one lost counts as a failed event above,
    # which is right, but the per-identity view shows it was a half failure.
    identities = identities_ok + identities_bad
    if identities:
        print(f"{'by identity':<12} {identities_ok:>9} {identities_bad:>7} "
              f"{identities_ok / identities:>6.1%}")

    crossings = by_type.get("crossing")
    if crossings:
        ok, bad = crossings["ok"], crossings["bad"]
        print(f"\nidentity survived {ok} of {ok + bad} crossings "
              f"({ok / (ok + bad):.1%})")

    if args.tracks:
        data = json.loads(args.tracks.read_text())
        frames = data["frames"]
        minutes = len(frames) / args.fps / 60
        print(f"\nover {len(frames)} frames ({minutes:.1f} min at {args.fps:g} fps):")
        print(f"  identity failures: {total_bad}")
        print(f"  failures per minute: {total_bad / minutes:.1f}")
        summary = data.get("summary", {})
        if summary:
            print(f"  tracker: {summary.get('tracker')}")
            print(f"  unique ids: {summary.get('unique_ids')}, "
                  f"mean lifetime {summary.get('mean_track_lifetime')} frames")

    print("\nThis is an audit of risk moments, not HOTA or IDF1. It answers "
          "whether identity held where it was tested, on the events found "
          "automatically. Report it as such.")


if __name__ == "__main__":
    main()
