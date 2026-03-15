import sentencepiece as spm
import sys

if __name__ == '___main__':
    sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
    with open(sys.argv[2], 'r') as rf, open(sys.argv[3], 'w') as wf:
        for line in rf:
            wf.write(' '.join(sp.encode(line.strip(), out_type=str) + '\n'))