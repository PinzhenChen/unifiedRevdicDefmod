import json
import numpy as np
from os import listdir
from os.path import isfile
from itertools import product

PREFIX = ""
SUFFIX = ""
LANGS = ["en", "es", "fr", "it", "ru"]
EMBEDS = ["sgns", "char", "electra"]


def merge_embeds(lang, prefix=PREFIX, suffix=SUFFIX, embeds=EMBEDS):
    data = []
    count = 0
    for embed in embeds:
        if embed == "electra" and lang in {"es", "it"}:
            continue

        filename = "ensemble-" + prefix + "-" + embed + "-" + lang + "-" + suffix
        if not isfile(filename):
            continue
        count += 1
        with open(filename) as f_in:
            items = json.load(f_in)
            for index, item in enumerate(items):
                for item in items:
                    try:
                        data[index]['id'] == item['id']
                    except:
                        data.append({'id': item['id']})
                    data[index][embed] = np.array(item[embed])

    with open("ensemble-" + prefix + "-" + "allembed" + "-" + lang + "-" + suffix,
              "w") as f_out:
        json.dump(data, f_out)
    print("For " + lang + ", we merged " + str(count) + " embeds.")


def ensemble(folders, prefix=PREFIX, suffix=SUFFIX, embeds=EMBEDS, langs=LANGS):
    for lang, embed in product(langs, embeds):
        print(lang, embed)

        if embed == "electra" and lang in {"es", "it"}:
            print("Invalid combination. Skipped " + embed + "-" + lang + ".")
            continue
        data = []
        count = 0

        begin = True
        for folder in folders:
            filenames = [folder + "/" + f for f in listdir(folder) if isfile(folder + "/" + f)]
            for filename in filenames:
                if not "-" + lang + "-" in filename or \
                    not "-" + embed + "-" in filename or \
                    not "-embedding-" in filename:
                    continue

                count += 1
                with open(filename) as f_in:
                    items = json.load(f_in)
                    for index, item in enumerate(items):
                        if begin:
                            data.append({'id': item['id'], embed: np.array(item[embed])})
                        else:
                            # print(data[index]['id'], item['id'])
                            # assert data[index]['id'] == item['id']
                            data[index][embed] += np.array(item[embed])

                begin = False
        for item in data:
            item[embed] = (item[embed] / count).tolist()

        with open("ensemble-" + prefix + "-" + embed + "-" + lang + "-" + suffix,
                  "w") as f_out:
            json.dump(data, f_out)

        print("For " + lang + "-" + embed + ", we ensembled " + str(count) + " models.")


if __name__ == '__main__':
    names = [
        "",
        ""
    ]
    folders = ["parent/dir/" + n for n in names]
    ensemble(folders=folders)
    for lang in LANGS:
        merge_embeds(lang=lang)

