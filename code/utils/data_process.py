import json
from transformers import BertTokenizer, BertModel
from multiprocessing import Pool
import torch
import gensim
import io


def merge_codwoe_all_lang():
    # unused
    for split in ['train', 'dev', 'trial.complete']:
        merged_data = []
        for lang in ['en', 'es', 'fr', 'it', 'ru']:
            with open("./data/" + lang + "." + split + ".json", "r") as f:
                merged_data.append(json.load(f))
        with open("./data/all." + split + ".json", "w") as f:
            json.dump(merged_data, f)


def process_tsinghua_data():
    with open('./data/others_data/tsinghua_data/data_dev.json', 'r') as f:
        data = json.load(f)
        new_data = []
        for d in data:
            d['gloss'] = d.pop('definitions')
            d.pop('lexnames')
            d.pop('root_affix')
            d.pop('sememes')
            new_data.append(d)
    with open('./data/others_data/ready/tsinghua_data/dev.json', 'w+') as f:
        json.dump(new_data, f)


def process_chang_data(path, outpath):
    data = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            word, context, definition = line.strip().split(";")
            d = {}
            d['word'] = word.strip()
            d['gloss'] = definition.strip()
            d['context'] = context.strip()
            data.append(d)
            line = f.readline()

    with open(outpath, 'w+') as f:
        json.dump(data, f)


def process_noraset_data(path, outpath):
    # unused
    data = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            word, _, _, definition = line.strip().split("\t")
            d = {}
            d['word'] = word.strip()
            d['gloss'] = definition.strip()
            data.append(d)
            line = f.readline()

    with open(outpath, 'w+') as f:
        json.dump(data, f)


def add_bert_embedding_from_def(file, file_out, split, lang='en', key='bert'):
    print("NOTE: This method computes BERT(word's definition) as a word embedding.")
    count = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.eval()
    with torch.no_grad():
        with open(file, 'r') as f:
            items = json.load(f)
            for item in items:
                gloss = item['gloss']
                inputs = tokenizer(gloss, return_tensors='pt')
                embedded = bert(**inputs)[1]  # pooler_output (shape (batch_size, hidden_size))
                assert list(embedded.size()) == [1, 768]
                embedding = embedded.squeeze().cpu().tolist()
                item[key] = embedding
                item['id'] = lang + '.' + split + '.' + str(count + 1)
                count += 1
                if count % 1000 == 0: print("processed {} files".format(count))

        with open(file_out, 'w') as f:
            json.dump(items, f)


def _sublist_ids(ls, sub_ls):
    l = len(sub_ls)
    for i in range(1, len(ls) - l):
        if ls[i:i + l] == sub_ls:
            return i, i + l


def add_bert_embedding_from_context(file, file_out, split, lang='en', key='bert'):
    print("NOTE: This method computes BERT(word|context) as a word embedding.")
    count = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.eval()
    with torch.no_grad():
        with open(file, 'r') as f:
            items = json.load(f)
            for item in items:
                context = item['context']
                inputs_ids = tokenizer(context)['input_ids']
                word_ids = tokenizer(item['word'])['input_ids'][1:-1]
                start_i, end_i = _sublist_ids(inputs_ids, word_ids)
                # assert inputs_ids[start_i:end_i] == word_ids
                inputs = tokenizer(context, return_tensors='pt')
                item[key] = torch.sum(bert(**inputs)[0][:, start_i : end_i, :],
                                      dim=1).squeeze().cpu().tolist()
                item['id'] = lang + '.' + split + '.' + str(count + 1)
                count += 1
                if count % 1000 == 0: print("processed {} files".format(count))

        with open(file_out, 'w') as f:
            json.dump(items, f)


def add_word2vec_embedding(file, file_out):
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/GoogleNews-vectors-negative300.bin', binary=True)
    print('w2v model loaded')

    count = 0
    missing_count = 0
    new_data = []
    with open(file, 'r') as f:
        items = json.load(f)
        for item in items:
            word = item['word']
            assert len(word.split()) == 1
            try:
                vector = model[word]
                item['sgns'] = vector.tolist()
                new_data.append(item)
            except KeyError:
                print('Key {} not present'.format(word))
                missing_count += 1
            count += 1
            if count % 1000 == 0: print("processed {} files".format(count))

    with open(file_out, 'w+') as f:
        json.dump(new_data, f)

    print("in total processed {} instances".format(count))
    print("there are {} words unable to find embedding from word2vec thus missing".format(
        missing_count))


def load_fasttext_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def add_fasttext_embedding(file, file_out):
    # Load fasttext vectors
    model = load_fasttext_vectors('./data/wiki-news-300d-1M.vec')
    print('fasttext model loaded')

    count = 0
    missing_count = 0
    new_data = []
    with open(file, 'r') as f:
        items = json.load(f)
        for item in items:
            word = item['word']
            assert len(word.split()) == 1
            try:
                vector = model[word]
                item['sgns'] = vector
                new_data.append(item)
            except KeyError:
                print('Key {} not present'.format(word))
                missing_count += 1
            count += 1
            if count % 1000 == 0: print("processed {} files".format(count))

    with open(file_out, 'w+') as f:
        json.dump(new_data, f)

    print("in total processed {} instances".format(count))
    print("there are {} words unable to find embedding from fasttext thus missing".format(
        missing_count))


def filter_hill_data(file, file_out):
    count = 0
    count_word = 0
    new_data = []
    with open(file, 'r') as f:
        items = json.load(f)
        for item in items:
            if 'see synonyms at' in item['gloss']:
                count_word += 1
            else:
                # only include instances with clean gloss
                new_data.append(item)
            count += 1
            if count % 5000 == 0: print("processed {} instances".format(count))

    with open(file_out, 'w') as f:
        json.dump(new_data, f)

    print("there are {} instances with 'see synonyms' as gloss".format(count_word))


def check_data(file):
    count_word = 0
    count_sgns = 0
    count_bert = 0
    count = 0
    with open(file, 'r') as f:
        items = json.load(f)
        for item in items:
            assert item['id'] != ''
            if item['word'] == '': count_word += 1
            if item['sgns'] == []: count_sgns += 1
            if item['bert'] == []: count_bert += 1
            count += 1
            if count % 2000 == 0: print("processed {} instances".format(count))

    print("there are {} word is empty".format(count_word))
    print("there are {} sgns is empty list".format(count_sgns))
    print("there are {} bert empty list".format(count_bert))


if __name__ == '__main__':
    pass
