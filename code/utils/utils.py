import json
import csv
import argparse

def _count_items(file):
    with open(file, 'r') as f:
        # print("Assuming that the item id in \"" \
        #       + file \
        #       + "\" is formatted as $lang.$split.....$number")
        # items = json.load(f)
        # assert (len(items) == int(items[-1]['id'].split('.')[-1])) or \
        #        (len(items) == int(items[-1]['id'].split('.')[-1]) + 1), \
        #        "len() and last id mismatch."
        return str(len(json.load(f)))


def add_track_to_output_id(file, track):
    with open(file, 'r') as f:
        items = json.load(f)
        for item in items:
            id_split = item['id'].split('.')
            item['id'] = ".".join([id_split[0], id_split[1], track, id_split[-1]])

    with open(file, 'w') as f:
        json.dump(items, f)


def change_gloss(file):
    pass
    # for inspection only
    with open(file, 'r') as f:
        items = json.load(f)
        for item in items:
            item['gloss'] = "the the the the the"

    with open(file, 'w') as f:
        json.dump(items, f)


def write_results_to_txt(dir, results, track):
    if track == 'revdict':
        keys = ['mse', 'cosine', 'ranking']
    elif track == 'defmod':
        keys = ['sense', 'lemma', 'mover']
    else:
        keys = None
    with open(dir + track + '_all_results.csv', 'w+') as f:
        for loss in keys:
            line = []
            for lang in ['en', 'es', 'fr', 'it', 'ru']:
                score = results[loss][lang]
                line += score
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([r for r in line])


def gather_results_in_one_dir(dir, model_name):
    revdict_results = {'mse': {'en': [], 'es': [], 'fr': [], 'it': [], 'ru': []},
                       'cosine': {'en': [], 'es': [], 'fr': [], 'it': [], 'ru': []},
                       'ranking': {'en': [], 'es': [], 'fr': [], 'it': [],
                                   'ru': []}}  # en es fr it ru # SGN char electra
    defmod_results = {'sense': {'en': [], 'es': [], 'fr': [], 'it': [], 'ru': []},
                      'lemma': {'en': [], 'es': [], 'fr': [], 'it': [], 'ru': []},
                      'mover': {'en': [], 'es': [], 'fr': [], 'it': [],
                                'ru': []}}  # en es fr it ru # SGN char electra

    for track in ['embedding', 'definition']:
        for lang in ['en', 'es', 'fr', 'it', 'ru']:
            for emb in ['sgns', 'char', 'electra']:
                file = dir + model_name + '-' + emb + '-' + lang + '-' + track + '-output.json.score.txt'
                try:
                    with open(file) as f:
                        line = f.readlines()
                        assert len(line) == 3
                        if track == 'embedding':
                            score = line[0].strip().split(':')[1]
                            revdict_results['mse'][lang].append(float(score))
                            score = line[1].strip().split(':')[1]
                            revdict_results['cosine'][lang].append(float(score))
                            score = line[2].strip().split(':')[1]
                            revdict_results['ranking'][lang].append(float(score))
                        else:
                            score = line[0].strip().split(':')[1]
                            defmod_results['mover'][lang].append(float(score))
                            score = line[1].strip().split(':')[1]
                            defmod_results['lemma'][lang].append(float(score))
                            score = line[2].strip().split(':')[1]
                            defmod_results['sense'][lang].append(float(score))
                except FileNotFoundError:
                    print('Warning!', file, "does not exist!!")

    print(revdict_results)
    print(defmod_results)
    write_results_to_txt(dir, revdict_results, 'revdict')
    write_results_to_txt(dir, defmod_results, 'defmod')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="demo script for participants")
    parser.add_argument("--file", type=str, default='exps/ae-baseline/outputs/autoencoder.json',
                        help="Location of json output file to be edited")
    parser.add_argument("--track", type=str, choices=("defmod", "revdict"),
                        help="the track to be added to the ids")
    parser.add_argument("--add_track", action='store_true',
                        help="add track to output json id field")
    parser.add_argument("--gather_results", action='store_true',
                        help="get all results and put in one file")
    parser.add_argument("--count_items", action='store_true',
                        help="count number of instances in a .json")

    args = parser.parse_args()

    if args.count_items:
        print("Assuming that the item id in \"" \
              + args.file \
              + "\" is formatted as $lang.$split.....$number")
        print("file " + args.file + " has " + _count_items(args.file) + " items.")

    if args.add_track:
        print("Assuming that the item id in \"" \
              + args.file \
              + "\" is formatted as $lang.$split.....$number,")
        print("the new id will be formatted as $lang.$split.$track.$number")
        add_track_to_output_id(args.file, args.track)
    if args.gather_results:
        gather_results_in_one_dir('some/directory/', 'model_name')
