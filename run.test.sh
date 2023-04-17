#! /bin/bash

BASEDIR=../../lab_resources/DDI/
#BASEDIR=../lab1/DDI/

# convert datasets to feature vectors
echo "Extracting features..."
python3 extract-features.py $BASEDIR/data/train/ > train.feat
python3 extract-features.py $BASEDIR/data/test/ > test.feat

# train CRF model
echo "Training CRF model..."
python3 train-crf.py model.crf < train.feat
# run CRF model
echo "Running CRF model..."
python3 predict.py model.crf < test.feat > test-CRF.out
# evaluate CRF results
echo "Evaluating CRF results..."
python3 evaluator.py NER $BASEDIR/data/test test-CRF.out > test-CRF.stats


#Extract Classification Features
cat train.feat | cut -f5- | grep -v ^$ > train.clf.feat


# train Naive Bayes model
echo "Training Naive Bayes model..."
python3 train-sklearn.py model.joblib vectorizer.joblib < train.clf.feat
# run Naive Bayes model
echo "Running Naive Bayes model..."
python3 predict-sklearn.py model.joblib vectorizer.joblib < test.feat > test-NB.out
# evaluate Naive Bayes results 
echo "Evaluating Naive Bayes results..."
python3 evaluator.py NER $BASEDIR/data/test test-NB.out > test-NB.stats

# remove auxiliary files.
#rm train.clf.feat
