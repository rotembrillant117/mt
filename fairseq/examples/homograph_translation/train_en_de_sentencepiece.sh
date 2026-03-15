#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
TOKENIZER_TYPE=bpe
SRC_BPE_TOKENS=8000
TGT_BPE_TOKENS=8000
SRC_DROPOUT=0.0
TGT_DROPOUT=0.0
SEED=0
DEVICE=0

EXPERIMENT_PREFIX="experiment"

while [[ "$#" -gt 0 ]]
do case $1 in
    --src-bpe-tokens) SRC_BPE_TOKENS=$2
    shift;;
    --tgt-bpe-tokens) TGT_BPE_TOKENS=$2
    shift;;
    --src-dropout) SRC_DROPOUT=$2
    shift;;
    --tgt-dropout) TGT_DROPOUT=$2
    shift;;
    --jamo-type) JAMO_TYPE=$2
    shift;;
    --tokenizer-type) TOKENIZER_TYPE=$2
    shift;;
    --seed) SEED=$2
    shift;;
    --device) DEVICE=$2
    shift;;
    --experiment-name) EXPERIMENT_PREFIX="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done
echo "========= PARAMETERS =========== "
echo -e "JAMO_TYPE $JAMO_TYPE \nSRC_TOKENS $SRC_BPE_TOKENS \nTGT_TOKENS $TGT_BPE_TOKENS \nSRC_DROPOUT $SRC_DROPOUT \nTGT_DROPOUT $TGT_DROPOUT \nSEED $SEED \nDEVICE $DEVICE \nNAME $EXPERIMENT_PREFIX\n"
echo "========= PARAMETERS =========== "


src=en
tgt=de
lang=en-de

EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_jamo_type_${JAMO_TYPE}_tokenizer_type_${TOKENIZER_TYPE}_BPE_${SRC_BPE_TOKENS}_${TGT_BPE_TOKENS}_dropout_${SRC_DROPOUT}_${TGT_DROPOUT}_seed_${SEED}.${lang}"

mkdir -p ../../${src}_${tgt}_sentencepiece_experiment_outputs

prep=experiments/$EXPERIMENT_NAME
tmp=$prep/tmp
orig=orig/${JAMO_TYPE}


if [ -d "../../${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}" ]
then
    echo "${EXPERIMENT_NAME} already done, SKIPPING"
    exit 0
fi

mkdir -p $prep

mkdir -p data-bin/$EXPERIMENT_NAME

BPE_CODE=$prep/code
BPE_VOCAB=$prep/vocab

python - <<PY
import sentencepiece as spm
spm.SentencePieceTrainer.Train(
    input="$orig/train.$src,$orig/train.$tgt",
    model_prefix="$prep/joint_tokenizer",
    vocab_size=int("$SRC_BPE_TOKENS"),
    model_type="$TOKENIZER_TYPE",
)
PY

python3 dump_unigram_vocab.py $prep/joint_tokenizer.vocab $prep/joint_tokenizer.dict

for f in train valid test; do
    echo "encode (joint tokenizer) ($src) to ${f}.${src}..."
    python - <<PY
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="$prep/joint_tokenizer.model")
with open("$orig/$f.$src", "r", encoding="utf-8") as fin, open("$prep/$f.$src", "w", encoding="utf-8") as fout:
    for line in fin:
        pieces = sp.encode(line.strip(), out_type=str)
        fout.write(" ".join(pieces) + "\n")
PY
    cp $prep/$f.$src data-bin/$EXPERIMENT_NAME/$f.$src

    echo "encode (joint tokenizer) ($tgt) to ${f}.${tgt}..."
    python - <<PY
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="$prep/joint_tokenizer.model")
with open("$orig/$f.$tgt", "r", encoding="utf-8") as fin, open("$prep/$f.$tgt", "w", encoding="utf-8") as fout:
    for line in fin:
        pieces = sp.encode(line.strip(), out_type=str)
        fout.write(" ".join(pieces) + "\n")
PY
    cp $prep/$f.$tgt data-bin/$EXPERIMENT_NAME/$f.$tgt
done

cd ../..

TEXT=examples/homograph_translation/experiments/$EXPERIMENT_NAME
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir examples/homograph_translation/data-bin/$EXPERIMENT_NAME \
    --workers 8 \
    --joined-dictionary \
    --srcdict examples/homograph_translation/experiments/$EXPERIMENT_NAME/joint_tokenizer.dict

cp ${TEXT}/joint_tokenizer.model ${TEXT}/joint_tokenizer.vocab ${TEXT}/joint_tokenizer.dict examples/homograph_translation/data-bin/$EXPERIMENT_NAME/
cp ${TEXT}/joint_tokenizer.dict examples/homograph_translation/data-bin/$EXPERIMENT_NAME/dict.${src}.txt
cp ${TEXT}/joint_tokenizer.dict examples/homograph_translation/data-bin/$EXPERIMENT_NAME/dict.${tgt}.txt


mkdir -p ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}/

CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-train  examples/homograph_translation/data-bin/$EXPERIMENT_NAME \
                                            --arch transformer_iwslt_de_en \
                                            --share-all-embeddings \
                                            --optimizer adam --adam-betas '(0.9, 0.98)' \
                                            --clip-norm 0.0 \
                                            --lr 5e-4 \
                                            --lr-scheduler inverse_sqrt \
                                            --warmup-updates 4000 \
                                            --validate-after-updates 1000 \
                                            --dropout 0.1 \
                                            --weight-decay 0.0001 \
                                            --criterion label_smoothed_cross_entropy \
                                            --label-smoothing 0.1 \
                                            --max-tokens 4096 \
                                            --eval-bleu  \
                                            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
                                            --eval-bleu-detok moses \
                                            --eval-bleu-detok-args '{"target_lang": "de"}' \
                                            --eval-bleu-remove-bpe=sentencepiece_${JAMO_TYPE} \
                                            --eval-bleu-print-samples \
                                            --best-checkpoint-metric bleu \
                                            --maximize-best-checkpoint-metric \
                                            --patience 8  \
                                            --save-dir "${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}" \
                                            --source-lang=$src \
                                            --target-lang=$tgt \
                                            --seed $SEED \
                                            --task "translation-with-subword-regularization" \
                                            --src-dropout $SRC_DROPOUT \
                                            --tgt-dropout $TGT_DROPOUT \
                                            --jamo-type $JAMO_TYPE \
                                            --bpe-impl-path "$(pwd)/examples/homograph_translation" \
                                            --raw-data-path "$(pwd)/examples/homograph_translation/orig/${JAMO_TYPE}" \
                                            --no-epoch-checkpoints > ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}/$EXPERIMENT_NAME.log
                                            


CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-generate examples/homograph_translation/data-bin/$EXPERIMENT_NAME \
                                        --path ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}/checkpoint_best.pt \
                                        --batch-size 128 \
                                        --beam 5 \
                                        --max-len-a 1.2 \
                                        --max-len-b 10 \
                                        --remove-bpe=sentencepiece_${JAMO_TYPE} \
                                        --source-lang=$src \
                                        --target-lang=$tgt \
                                        > ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}/bleu_unprocessed.log



cd ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}

grep --text ^H bleu_unprocessed.log | cut -f3- > gen.out.sys
grep --text ^T bleu_unprocessed.log | cut -f2- > gen.out.ref
cat gen.out.sys | sacremoses -l de detokenize > gen.out.sys.detok
cat gen.out.ref | sacremoses -l de detokenize > gen.out.ref.detok
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m bleu -b -w 4 > BLEU.txt
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m chrf -b > CHRF.txt
