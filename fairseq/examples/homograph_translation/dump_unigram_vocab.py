import sys
if __name__ == '__main__':
    unigram_file = open(sys.argv[1], 'r')
    output_file = open(sys.argv[2], 'w')

    for idx, line in enumerate(unigram_file):

        if len(line.split()) == 2 and len(line.strip()) > 0 and all(w not in line for w in ["<unk>", "<s>", "</s>"]):
            v, _ = line.strip().split()
            output_file.write(f"{v.strip()} {idx + 1000}\n")
    unigram_file.close(), output_file.close()