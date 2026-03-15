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


src=je
tgt=ko
lang=je-ko

EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_jamo_type_${JAMO_TYPE}_tokenizer_type_${TOKENIZER_TYPE}_BPE_${SRC_BPE_TOKENS}_${TGT_BPE_TOKENS}_dropout_${SRC_DROPOUT}_${TGT_DROPOUT}_seed_${SEED}.${lang}"

mkdir ../../${src}_${tgt}_sentencepiece_experiment_outputs

prep=experiments/$EXPERIMENT_NAME
tmp=$prep/tmp
orig=orig/${JAMO_TYPE}


if [ -d "../../${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}" ]
then
    echo "${EXPERIMENT_NAME} already done, SKIPPING"
    exit 0
fi

mkdir $prep

mkdir data-bin/$EXPERIMENT_NAME

BPE_CODE=$prep/code
BPE_VOCAB=$prep/vocab

# echo "learn_BPE for src: $src"
# python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $orig/train.$src -s $SRC_BPE_TOKENS -t -o $BPE_CODE.$src --write-vocabulary $BPE_VOCAB.$src

# echo "learn_BPE for tgt: $tgt"
# python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $orig/train.$tgt -s $TGT_BPE_TOKENS -t -o $BPE_CODE.$tgt --write-vocabulary $BPE_VOCAB.$tgt

spm_train --input=$orig/train.$src --model_prefix=$prep/${src}_tokenizer --vocab_size=$SRC_BPE_TOKENS --split_by_whitespace false --character_coverage=1.0 --normalization_rule_name="identity" --model_type=${TOKENIZER_TYPE}
spm_train --input=$orig/train.$tgt --model_prefix=$prep/${tgt}_tokenizer --vocab_size=$TGT_BPE_TOKENS --split_by_whitespace false --character_coverage=1.0 --normalization_rule_name="identity" --model_type=${TOKENIZER_TYPE}

python3 dump_unigram_vocab.py $prep/${src}_tokenizer.vocab $prep/${src}_tokenizer.dict
python3 dump_unigram_vocab.py $prep/${tgt}_tokenizer.vocab $prep/${tgt}_tokenizer.dict

# for f in train valid test; do
#     echo "apply_bpe.py ($src) to ${f}.${src}..."
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE.$src --vocabulary $BPE_VOCAB.$src < $orig/$f.$src > $prep/$f.$src
#     cp $prep/$f.$src data-bin/$EXPERIMENT_NAME/$f.$src

#     echo "apply_bpe.py ($tgt) to ${f}.${tgt}..."
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE.$tgt --vocabulary $BPE_VOCAB.$tgt < $orig/$f.$tgt > $prep/$f.$tgt
#     cp $prep/$f.$tgt data-bin/$EXPERIMENT_NAME/$f.$tgt
# done

for f in train valid test; do
    echo "apply_bpe.py ($src) to ${f}.${src}..."
    spm_encode --model $prep/${src}_tokenizer.model --output_format=piece --input=$orig/$f.$src --output=$prep/$f.$src
    cp $prep/$f.$src data-bin/$EXPERIMENT_NAME/$f.$src

    echo "apply_bpe.py ($tgt) to ${f}.${tgt}..."
    spm_encode --model $prep/${tgt}_tokenizer.model --output_format=piece --input=$orig/$f.$tgt --output=$prep/$f.$tgt
    cp $prep/$f.$tgt data-bin/$EXPERIMENT_NAME/$f.$tgt
done


cd ../..

TEXT=examples/kr_translation/experiments/$EXPERIMENT_NAME
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir examples/kr_translation/data-bin/$EXPERIMENT_NAME \
    --workers 8 \
    --srcdict examples/kr_translation/experiments/$EXPERIMENT_NAME/${src}_tokenizer.dict \
    --tgtdict examples/kr_translation/experiments/$EXPERIMENT_NAME/${tgt}_tokenizer.dict

# cp $orig/train.$src examples/kr_translation/data-bin/$EXPERIMENT_NAME/train.raw.$src
# cp $orig/train.$tgt examples/kr_translation/data-bin/$EXPERIMENT_NAME/train.raw.$tgt

cp ${TEXT}/${src}_tokenizer.model ${TEXT}/${src}_tokenizer.vocab ${TEXT}/${src}_tokenizer.dict examples/kr_translation/data-bin/$EXPERIMENT_NAME/
cp ${TEXT}/${tgt}_tokenizer.model ${TEXT}/${tgt}_tokenizer.vocab ${TEXT}/${tgt}_tokenizer.dict examples/kr_translation/data-bin/$EXPERIMENT_NAME/

# sed -i -r 's/(@@ )|(@@ ?$)//g' examples/kr_translation/data-bin/$EXPERIMENT_NAME/train.raw.$src
# sed -i -r 's/(@@ )|(@@ ?$)//g' examples/kr_translation/data-bin/$EXPERIMENT_NAME/train.raw.$tgt

mkdir ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}/

CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-train  examples/kr_translation/data-bin/$EXPERIMENT_NAME \
                                            --arch transformer_iwslt_de_en \
                                            --share-decoder-input-output-embed \
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
                                            --eval-bleu-detok-args '{"target_lang": "ko"}' \
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
                                            --bpe-impl-path "/home/cognetta-m/github/jamo_bpe_paper/fairseq/examples/kr_translation/" \
                                            --raw-data-path "/home/cognetta-m/github/jamo_bpe_paper/fairseq/examples/kr_translation/orig/${JAMO_TYPE}" \
                                            --no-epoch-checkpoints > ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}/$EXPERIMENT_NAME.log
                                            


CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-generate examples/kr_translation/data-bin/$EXPERIMENT_NAME \
                                        --path ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}/checkpoint_best.pt \
                                        --batch-size 128 \
                                        --beam 5 \
                                        --max-len-a 1.2 \
                                        --max-len-b 10 \
                                        --remove-bpe=sentencepiece_${JAMO_TYPE} > ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}/bleu_unprocessed.log



cd ${src}_${tgt}_sentencepiece_experiment_outputs/${EXPERIMENT_NAME}

grep --text ^H bleu_unprocessed.log | cut -f3- > gen.out.sys
grep --text ^T bleu_unprocessed.log | cut -f2- > gen.out.ref
cat gen.out.sys | sacremoses -l ko detokenize  > gen.out.sys.detok
cat gen.out.ref | sacremoses -l ko detokenize  > gen.out.ref.detok
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m bleu -b -w 4 > BLEU.txt
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m chrf -b > CHRF.txt
