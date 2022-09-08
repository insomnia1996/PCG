import random
domains=["humans","books","songs"]
N=100
for domain in domains:
    with open("./{}/original_data/sample_source.box".format(domain),'r') as f:
        line = f.readline()
        boxlines = []
        while line:
            line = f.readline().strip('\n')
            if line:
                boxlines.append(line)
    with open("./{}/original_data/sample_source.summary".format(domain),'r') as f:
        line = f.readline()
        sumlines = []
        while line:
            line = f.readline().strip('\n')
            if line:
                sumlines.append(line)

    assert len(sumlines)==len(boxlines)
    idxlst=[i for i in range(len(sumlines))]
    idsample = random.sample(idxlst, N)

    with open("./{}/original_data/train.box".format(domain),'w') as f:
        for idx in idsample:
            f.write(boxlines[idx]+'\n')
    with open("./{}/original_data/train.summary".format(domain),'w') as f:
        for idx in idsample:
            f.write(sumlines[idx]+'\n')
