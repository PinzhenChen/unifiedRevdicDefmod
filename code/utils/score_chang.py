# Adapted from https://github.com/MiuLab/GenDef/blob/master/get_bleu_rouge.py

import nltk.translate.bleu_score as bleu_score
from rouge import Rouge
import sys
import argparse

def get_parser(
        parser=argparse.ArgumentParser(
            description="Please provide an output file and a reference file."
        ),
):
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path to the model output file"
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        help="path to the reference file"
    )
    return parser
args = get_parser().parse_args()

rouge = Rouge()

myAnswers = []
references = []
with open(args.output) as f_out, open(args.reference) as f_ref:
    for o, r in zip(f_out, f_ref):
        myAnswers.append(o.strip())
        references.append(r.strip())
assert len(myAnswers) == len(references)

BLEUscore = bleu_score.corpus_bleu([[r.split()] for r in references], [my.split() for my in myAnswers])
Rougescore = rouge.get_scores(myAnswers, references, avg=True, ignore_empty=True)
print('[Max 100] BLEU: {:.1f}, ROUGE-L:F {:.1f}\n'.format(100*BLEUscore, 100*Rougescore['rouge-l']['f']))
