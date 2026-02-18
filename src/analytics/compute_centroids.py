import csv
from collections import defaultdict

INPUT_CSV = "tracks.csv"
OUTPUT_CSV = "team_centroids.csv"

def main() -> None:
    # key = (frame, team_id) -> [sum_cx, sum_cy, count]
    agg = defaultdict(lambda: [0.0, 0.0, 0.0])

    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)

        # expect columns: frame, team_id, cx, cy (plus others)
        for row in r:
            team_id = int(row["team_id"])
            if team_id == -1:
                continue

            frame = int(row["frame"])
            cx = float(row["cx"])
            cy = float(row["cy"])

            key = (frame, team_id)
            agg[key][0] += cx
            agg[key][1] += cy
            agg[key][2] += 1
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame","team_id","centroid_x","centroid_y","n_players"])

        for (frame, team_id) in sorted(agg.keys()):
            sum_cx, sum_cy, n = agg[(frame, team_id)]
            centroid_x = sum_cx / n
            centroid_y = sum_cy / n
            w.writerow([frame, team_id, centroid_x, centroid_y, n])

    print(f"Saved:{OUTPUT_CSV}")

if __name__ == "__main__":
    main()