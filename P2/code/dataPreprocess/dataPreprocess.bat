@echo off


SET TRAIN_FILE=D:/senior/aiCourse/dataSource/comment_classification/train/sentiment_analysis_trainingset.csv
SET VALIDATION_FILE=D:/senior/aiCourse/dataSource/comment_classification/validation/sentiment_analysis_validationset.csv
SET TESTA_FILE=D:/senior/aiCourse/dataSource/comment_classification/test/sentiment_analysis_testa.csv


SET TRAIN_FILE_OUTPUT=D:/senior/aiCourse/dataSource/comment_classification/output/train.json
SET VALIDATION_FILE_OUTPUT=D:/senior/aiCourse/dataSource/comment_classification/output/validation.json
SET TESTA_FILE_OUTPUT=D:/senior/aiCourse/dataSource/comment_classification/output/testa.json


SET VOCAB_FILE=D:/senior/aiCourse/dataSource/comment_classification/output/vocab.txt
SET VOCAB_SIZE=50000


SET EMBEDDING_FILE=D:/senior/aiCourse/dataSource/comment_classification/sgns.wiki.word
SET EMBEDDING_FILE_OUTPUT=D:/senior/aiCourse/dataSource/comment_classification/embedding/embedding.txt

ECHO %TRAIN_FILE%


echo 'Process training file ...'
python dataLoader.py --data_file=%TRAIN_FILE% --output_file=%TRAIN_FILE_OUTPUT% --vocab_file=%VOCAB_FILE% --vocab_size=%VOCAB_SIZE%

echo 'Process testa file ...'
python dataLoader.py --data_file=%TESTA_FILE% --output_file=%TESTA_FILE_OUTPUT% 

echo 'Process validation file ...'
python dataLoader.py --data_file=%VALIDATION_FILE% --output_file=%VALIDATION_FILE_OUTPUT% 

::echo 'Get pretrained embedding ...'
::python dataLoader.py --data_file=%EMBEDDING_FILE% --output_file=%EMBEDDING_FILE_OUTPUT% --vocab_file=%VOCAB_FILE% --embedding=True

pause