import codecs
import sys
from bert_serving.client import BertClient
bc = BertClient()


word_dict_path = sys.argv[1]

word_embedding_path = sys.argv[2]
lines = [line.strip().split("\t")[0] for line in codecs.open(word_dict_path, "r", 'utf-8').readlines()]
count = 0
with codecs.open(word_embedding_path, "w", 'utf-8') as f_out:
    f_out.write('{} {}\n'.format(len(lines), 768))
    for line in lines:
        vector = " ".join(str(round(item,6)) for item in bc.encode([line]).tolist()[0])
        write_line = '{} {}\n'.format(str(line), vector)
        f_out.write(write_line)
        count += 1
        if count%10000 == 0:
            print("processed {} line".format(count))