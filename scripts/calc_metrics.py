import argparse
import math
import os
from os.path import join
from pickle import load


METRIC_KEYS = {
    "CR": "collision_rate",
    "OS": "final_score",
    "RR": "avg_red_light_freq",
    "SS": "avg_stop_sign_freq",
    "OR": "out_of_road_length",
    "RF": "route_following_stability",
    "Comp": "route_completion",
    "TS": "avg_time_spent",
    "ACC": "avg_acceleration",
    "YV": "avg_yaw_velocity",
    "LI": "avg_lane_invasion_freq",
}


def parse_int_list(value):
    if value is None:
        return None
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True)
    parser.add_argument("--behaviors", type=str, default="1-5")
    parser.add_argument("--routes", type=str, default="0-9")
    parser.add_argument("--missing", choices=["skip", "error"], default="skip")
    return parser


def load_pickle(path):
    with open(path, "rb") as f:
        return load(f)


def main():
    args = build_parser().parse_args()
    behaviors = parse_int_list(args.behaviors)
    routes = parse_int_list(args.routes)

    totals = {name: 0.0 for name in METRIC_KEYS}
    ade_total = 0.0
    ade_count = 0
    result_count = 0

    for behavior in behaviors:
        for route in routes:
            stem = f"OPT_behavior_{behavior}_opt_ROUTE-{route}"
            result_file = join(args.folder, f"{stem}_results.pkl")
            record_file = join(args.folder, f"{stem}_records.pkl")

            if not os.path.exists(result_file) or not os.path.exists(record_file):
                if args.missing == "error":
                    raise FileNotFoundError(f"Missing result or record file for {stem}")
                print(f"Skipping missing files for {stem}")
                continue

            results = load_pickle(result_file)
            records = load_pickle(record_file)
            result_count += 1

            for display_name, key in METRIC_KEYS.items():
                totals[display_name] += results[key]

            for record_values in records.values():
                for value in record_values:
                    ego_x = value["ego_x"]
                    ego_y = value["ego_y"]
                    adv_x = value["adv_agent_0"]["x"]
                    adv_y = value["adv_agent_0"]["y"]
                    ade_count += 1
                    ade_total += math.sqrt((ego_x - adv_x) ** 2 + (ego_y - adv_y) ** 2)

    if result_count == 0:
        raise RuntimeError("No result files were loaded.")

    for display_name in METRIC_KEYS:
        print(f"{display_name}: {totals[display_name] / result_count}")
    if ade_count:
        print(f"ADE: {ade_total / ade_count} with count={ade_count}")
    else:
        print("ADE: n/a with count=0")


if __name__ == "__main__":
    main()
