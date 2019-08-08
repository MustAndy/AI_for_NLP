TRAIN_FILE=/home/wangjq/Desktop/aicourse/dataSource/comment-classification/train/train.csv
VALIDATION_FILE=/home/wangjq/Desktop/aicourse/dataSource/comment-classification/validation/validation.csv
TEST_FILE=/home/wangjq/Desktop/aicourse/dataSource/comment-classification/test/test.csv
#TESTB_FILE=/data/xueyou/data/ai_challenger_sentiment/ai_challenger_sentimetn_analysis_testb_20180816/sentiment_analysis_testb.csv

# Path to pretrained embedding file
EMBEDDING_FILE=/home/wangjq/Desktop/aicourse/dataSource/comment-classification/sgns.wiki.word

TRAIN_FILE_OUTPUT=/home/wangjq/Desktop/aicourse/dataSource/comment-classification/output/train.json
VALIDATION_FILE_OUTPUT=/home/wangjq/Desktop/aicourse/dataSource/comment-classification/output/validation.json
TEST_FILE_OUTPUT=/home/wangjq/Desktop/aicourse/dataSource/comment-classification/output/test.json
EMBEDDING_FILE_OUTPUT=/home/wangjq/Desktop/aicourse/dataSource/comment-classification/output/embedding.txt
VOCAB_FILE=/home/wangjq/Desktop/aicourse/dataSource/comment-classification/output/vocab.txt

VOCAB_SIZE=50000


echo 'Process training file ...'
#python3 dataLoader.py \
    #--data_file=$TRAIN_FILE \
   # --output_file=$TRAIN_FILE_OUTPUT \
  #  --vocab_file=$VOCAB_FILE \
 #   --vocab_size=$VOCAB_SIZE

#echo 'Process validation file ...'
#python3 dataLoader.py \
  #  --data_file=$VALIDATION_FILE \
 #   --output_file=$VALIDATION_FILE_OUTPUT

#echo 'Process testa file ...'
#python3 dataLoader.py \
 #   --data_file=$TESTA_FILE \
#    --output_file=$TEST_FILE_OUTPUT

# Uncomment following code to get testb file
# echo 'Process testb file ...'
# python data_preprocess.py \
#     --data_file=$TESTB_FILE \
#     --output_file=data/testb.json

echo 'Get pretrained embedding ...'
python3 dataLoader.py \
    --data_file=$EMBEDDING_FILE \
    --output_file=$EMBEDDING_FILE_OUTPUT \
    --vocab_file=$VOCAB_FILE \
    --embedding=True


