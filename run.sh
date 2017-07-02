M="scratch"
O="output_scratch"
MAXLEN=12    # NOTICE: if you want to modify this value, you need also modify MAXLEN in skip_thoughts_model.py
GIVEN=10


model_dir=$M/translate_l${MAXLEN}_att_h
output_dir=$O/traslate_l${MAXLEN}_att_h
#RELOAD_MODEL_ALL=$M/translate_l${MAXLEN}_att_h/preG_model-40000

mkdir $M
mkdir $O

mkdir ${model_dir}
mkdir ${output_dir}

cp -r skip_thoughts ${model_dir}
cp run.sh ${model_dir}
cp train.py ${model_dir}

python train.py --train_dir=${model_dir}  \
                --maxlen ${MAXLEN} \
                --given_num ${GIVEN} \
                --pretrain_G_steps 0 \
                --pretrain_D_steps 0 \
                --mixer_period 2500 \
                --mixer_step 2 \
                --train_corpus_en data/short.en \
                --train_corpus_fr data/short.fr \
                #--reload_model_all ${RELOAD_MODEL_ALL} 

