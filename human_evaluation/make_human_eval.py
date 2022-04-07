import random
import csv

FILES=[
"test_word.txt",
"test_def.txt",
"transformer_baseline.txt",
"unified.txt",
"unified_share.txt"]


with open(FILES[1]) as f_ref, open(FILES[2]) as f_base, open(FILES[4]) as f_unified, open("eval_ref.csv", "w", newline='') as eval, open("ans_ref.csv", "w") as gold:
    writer = csv.writer(eval)
    data = []
    for r, d, u in zip(f_ref, f_base, f_unified):
        data.append([r.strip(), d.strip(), u.strip()])
    samples = random.sample(data, k=80)

    for sample in samples:
        if random.random() > 0.5:
            writer.writerow([sample[0], sample[1], sample[2]])
            gold.write("2\n")
        else:
            writer.writerow([sample[0], sample[2], sample[1]])
            gold.write("1\n")

with open(FILES[0]) as f_ref, open(FILES[2]) as f_base, open(FILES[4]) as f_unified, open("eval_word.csv", "w", newline='') as eval, open("ans_word.csv", "w") as gold:
    writer = csv.writer(eval)
    data = []
    for r, d, u in zip(f_ref, f_base, f_unified):
        data.append([r,d,u])
    samples = random.sample(data, k=80)

    for sample in samples:
        if random.random() > 0.5:
            writer.writerow([sample[0], sample[1], sample[2]])
            gold.write("2\n")
        else:
            writer.writerow([sample[0], sample[2], sample[1]])
            gold.write("1\n")
