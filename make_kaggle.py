nums, texts, labels = [], [], []

with open("kaggle-irony.csv", 'r') as f:
    i = 0
    for line in f:
        if i == 0:
            nums.append(line.rstrip())
            i += 1
        elif i == 1:
            texts.append(line.rstrip())
            i += 1
        else:
            labels.append(
                {"-1": "0", "1": "1"}[line.rstrip()]
            )
            i = 0

content = "\t".join(["Tweet index", "Label", "Tweet text\n"])

with open("kaggle-irony.tsv", 'w') as k:
    for line in zip(nums, labels, texts):
        content += "\t".join(line) + "\n"
    k.write(content)
