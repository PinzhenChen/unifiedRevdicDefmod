import json
import argparse

def get_parser(
        parser=argparse.ArgumentParser(
            description="Please provide an input file."
        ),
):
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path to the input file"
    )
    return parser

args = get_parser().parse_args()

with open(args.input) as in_file, open(args.input+".txt", "w") as out_file:
    items = json.load(in_file)
    for item in items:
        out_file.write(item["gloss"] + "\n")
