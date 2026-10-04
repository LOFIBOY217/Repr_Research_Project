import argparse
import json
from recon_fd.diagnostics.trajectory import compare_evaluations
from recon_fd.provenance import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen fixed-evaluator reconstruction trajectories")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--heldout", nargs="+", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_evaluations(args.metrics, args.target, args.heldout, args.tolerance)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
