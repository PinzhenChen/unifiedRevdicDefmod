import csv

with open("ans_ref.csv") as f_ans_ref, open("ans_word.csv") as f_ans_word, \
     open("eval_ref.csv") as f_human_ref, open("eval_word.csv") as f_human_word:

    reader = csv.reader(f_ans_ref, delimiter=',')
    ans_ref = [row[0] for row in reader]

    reader = csv.reader(f_ans_word, delimiter=',')
    ans_word = [row[0] for row in reader]

    reader = csv.reader(f_human_ref, delimiter=',')
    human_ref = [row[3] for row in reader]

    reader = csv.reader(f_human_word, delimiter=',')
    human_word = [row[3] for row in reader]

    zeros_ref = sum([n == "0" for n in human_ref])
    zeros_word = sum([n == "0" for n in human_word])

    unified_ref = sum([n != "0" and n == r for n,r in zip(human_ref, ans_ref)])
    base_ref = sum([n != "0" and n != r for n,r in zip(human_ref, ans_ref)])

    unified_word = sum([n != "0" and n == r for n,r in zip(human_word, ans_word)])
    base_word = sum([n != "0" and n != r for n,r in zip(human_word, ans_word)])

    #assert unified_ref + base_ref + zeros_ref == 80
    #assert unified_word + base_word + zeros_word == 80
    print("count: unified, baseline, same_output")
    print("reference-based:")
    print(unified_ref, base_ref, zeros_ref)
    print("reference-less:")
    print(unified_word, base_word, zeros_word)

