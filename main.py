# Support libs
import copy
import os
import random
import time
import warnings

import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.model_selection as ms
from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from confens.classifiers.ConfidenceEnsemble import ConfidenceEnsemble
from confens.utils.classifier_utils import get_classifier_name
from sklearn.base import is_classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

# --------------------------------------- GLOBAL VARS ------------------------------------------------

# Name of the input folder containing datasets in CSV format
CSV_FOLDER = "datasets"
# Name of the output folder
OUT_FOLDER = "output"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "result_metrics.csv"
# True if CSVs of each ensemble have to be printed
PRINT_CSV = True
# Percentage of test data wrt train data
TT_SPLIT = 0.5
# To be set if only a subset of rows of dataset have to be used
LIMIT_ROWS = 100000

# Set random seed for reproducibility of experiments
random.seed(42)
numpy.random.seed(42)


# ---------------------------------------- SUPPORT FUNCTIONS -----------------------------------------------

def print_dataframe(dataset_name, clf_name, clf, x, y,
                    tt_tag, unique_classes, diversity_tag) -> None:
    """
    Prints a CSV file containing dataset features, true label, ensemble prediction and predictions of base learners
    :param dataset_name: name of the dataset to be printed
    :param clf_name: name of the classifier
    :param clf: classifier object
    :param x: test set
    :param y: test labels
    :param tt_tag: TRAIN/TEST tag
    :param unique_classes: list of unique classes of the problem
    :param diversity_tag: string tag for diversity of ensembles
    :return:
    """
    out_df = copy.deepcopy(x)
    out_df.columns = ["inputfeature[" + col + "]" for col in out_df.columns]
    out_df["true_label"] = y
    out_df["ensemble_probabilities"] = [";".join([str(a) for a in x]) for x in clf.predict_proba(x.to_numpy())]
    out_df["ensemble_predicted_label"] = clf.predict(x.to_numpy())
    if hasattr(clf, "estimators_"):
        if isinstance(clf, ConfidenceEnsemble):
            # This is specific for Confidence Ensembles
            probas = clf.predict_proba(x.to_numpy(), get_base=True)
            preds = clf.predict(x.to_numpy(), get_base=True)
            i = 0
            for channel_name in preds[1].keys():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out_df[channel_name + "_probas"] = [";".join([str(a) for a in x]) for x in probas[1][channel_name]]
                    out_df[channel_name + "_pred"] = unique_classes[numpy.asarray(preds[1][channel_name], dtype=int)]
                i += 1
        else:
            # This is for other ensembles
            i = 0
            for channel in clf.estimators_:
                ch_tag = "channel" + str(i) + "_" + get_classifier_name(channel)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out_df[ch_tag + "_probas"] = [";".join([str(a) for a in x]) for x in
                                                  channel.predict_proba(x.to_numpy())]
                    out_df[ch_tag + "_pred"] = unique_classes[numpy.asarray(channel.predict(x.to_numpy()), dtype=int)]
                    i += 1
    out_df.to_csv(os.path.join(OUT_FOLDER, diversity_tag + "_" + dataset_name.replace(".csv", "") +
                               "_" + clf_name + "_" + tt_tag + ".csv"),
                  index=False)


def current_milli_time() -> int:
    """
    gets the current time in ms
    :return: a long int
    """
    return round(time.time() * 1000)


def get_base_learners() -> list:
    """
    Function to get base (simple) learners
    :return: the list of classifiers to be trained
    """
    return [
        # Single
        Pipeline([("norm", MinMaxScaler()), ("gnb", GaussianNB())]),
        LinearDiscriminantAnalysis(),
        LogisticRegression(tol=1e-3),
        DecisionTreeClassifier(),
        ExtraTreeClassifier(),
        # Bagging
        RandomForestClassifier(n_estimators=100),
        ExtraTreesClassifier(n_estimators=100),
        # Boosting
        AdaBoostClassifier(n_estimators=100),
        XGBClassifier(),
        GradientBoostingClassifier(n_estimators=100),
    ]


def get_base_couples():
    """
    Same as the "get_base_learners", but in a format that can be used by scikit-learn's VotingClassifier
    :return: the list of classifiers to be trained
    """
    return [
        # Single
        ('gnb', Pipeline([("norm", MinMaxScaler()), ("gnb", GaussianNB())])),
        ('lda', LinearDiscriminantAnalysis()),
        ('lr', LogisticRegression(tol=1e-3)),
        ('dt', DecisionTreeClassifier()),
        ('et', ExtraTreeClassifier()),
        # Bagging
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('ets', ExtraTreesClassifier(n_estimators=100)),
        # Boosting
        ('ada', AdaBoostClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(n_estimators=100)),
        ('gbc', GradientBoostingClassifier(n_estimators=100)),
    ]


def print_performance(classifier, c_name, x, y, train_time, tt_tag, diversity_tag) -> None:
    """
    Function that prints a row to CSV file the dataset and individual predictions of ensemble and base-learners
    :param classifier: the classifier under study
    :param c_name: the name of the classifier
    :param x: test set
    :param y: test labels
    :param train_time: the time needed to train the classifier
    :param tt_tag: train/test tag to be printed
    :param diversity_tag: tag that specifies the type of ensemble diversity
    """
    # Scoring
    start_time = current_milli_time()
    y_pred = classifier.predict(x.to_numpy())
    test_time = current_milli_time() - start_time

    # Computing Classification Performance Metrics
    acc = metrics.accuracy_score(y, y_pred)
    misc = int((1 - acc) * len(y))
    mcc = abs(metrics.matthews_corrcoef(y, y_pred))
    bacc = metrics.balanced_accuracy_score(y, y_pred)

    # Prints just accuracy for multi-class classification problems, no confusion matrix
    print('%s Accuracy: %.3f, MCC: %.3f, train time: %d \t-> %s' %
          (tt_tag, acc, mcc, train_time, c_name))

    # Updates CSV file form metrics of experiment
    with open(os.path.join(OUT_FOLDER, SCORES_FILE), "a") as myfile:
        # Prints result of experiment in CSV file
        myfile.write(full_name + "," + c_name + "," +
                     str(TT_SPLIT) + "," + str(tt_tag) + ',' + str(diversity_tag) +
                     ',' + str(acc) + "," + str(misc) + "," + str(mcc) + "," +
                     str(bacc) + "," + str(train_time) + "," + str(test_time) + "\n")


def get_data_diversity_learners() -> list:
    """
    Gets data diversity learners
    :return a classifier list
    """
    return [
        ExtraTreesClassifier(n_estimators=100),
        RandomForestClassifier(n_estimators=100),
        ConfidenceBagging(clf=ExtraTreeClassifier(), n_base=100, conf_thr=0.8, parallel_train=True),
        ConfidenceBoosting(clf=ExtraTreeClassifier(), n_base=100, conf_thr=0.8, learning_rate=3)
    ]


def get_alg_diversity_learners() -> list:
    """
    Gets alg diversity learners
    :return list of classifiers
    """
    base = get_base_learners()
    return [
        VotingClassifier(estimators=get_base_couples(), voting='soft'),
        ConfidenceBagging(clf=base, n_base=len(base), conf_thr=0.8, parallel_train=True),
        ConfidenceBoosting(clf=base, n_base=len(base), conf_thr=0.8, learning_rate=3)
    ]


def get_data_alg_diversity_learners() -> list:
    """
    Gets the ensembles that are diverse from both a data and algorithm viewpoint
    :return: a classifier list
    """
    base = get_base_learners()
    return [
        ConfidenceBagging(clf=base, n_base=100, conf_thr=0.8, parallel_train=True),
        ConfidenceBoosting(clf=base, n_base=100, conf_thr=0.8, learning_rate=3)
    ]


def exercise_classifier_batch(dataset_name, clf_list, tag, print_csv, x_train, y_train,
                              x_test, y_test, unique_classes, existing_exps) -> None:
    """
    Exercises a batch of classifiers
    :param dataset_name: the name of the dataset
    :param clf_list: list of classifiers to be exercised
    :param tag: diversity tag
    :param print_csv: True if CSV dataframe with base-learner scores have to be printed
    :param x_train: train set
    :param y_train: train labels
    :param x_test: test set
    :param y_test: test labels
    :param unique_classes: list of unique class labels
    :param existing_exps: dataframe containing exiting experiments
    :return:
    """
    # Loop for training and testing each learner
    for clf in clf_list:

        # Getting classifier name
        c_name = get_classifier_name(clf)
        c_name = c_name if c_name != "Pipeline" else clf.named_steps[1]

        # This is to check if the result has already been computed
        if existing_exps is not None and (((existing_exps['dataset_tag'] == full_name) &
                                           (existing_exps['clf'] == c_name) &
                                           (existing_exps['div_tag'] == tag)).any()):
            print('Skipping classifier %s, already in the results' % c_name)

        elif is_classifier(clf):
            start_ms = current_milli_time()
            clf.fit(x_train.to_numpy(), y_train)
            train_time = current_milli_time() - start_ms

            if print_csv:
                # Print CSVs
                print_dataframe(dataset_name, c_name, clf, x_train, y_train, "TRAIN", unique_classes, tag)
                print_dataframe(dataset_name, c_name, clf, x_test, y_test, "TEST", unique_classes, tag)

            print_performance(clf, c_name, x_train, y_train, train_time, "TRAIN", tag)
            print_performance(clf, c_name, x_test, y_test, train_time, "TEST", tag)


# ----------------------- MAIN ROUTINE ---------------------
if __name__ == '__main__':

    existing_exps = None
    if os.path.exists(os.path.join(OUT_FOLDER, SCORES_FILE)):
        existing_exps = pandas.read_csv(os.path.join(OUT_FOLDER, SCORES_FILE))
        existing_exps = existing_exps.loc[:, ['dataset_tag', 'clf', 'div_tag']]
    else:
        with open(os.path.join(OUT_FOLDER, SCORES_FILE), 'w') as f:
            f.write("dataset_tag,clf,tt_split,train/test,div_tag,acc,misc,mcc,bacc,train_time,test_time\n")

    # Iterating over CSV files in folder
    for dataset_file in os.listdir(CSV_FOLDER):
        full_name = os.path.join(CSV_FOLDER, dataset_file)
        if full_name.endswith(".csv"):

            # if file is a CSV, it is assumed to be a dataset to be processed
            df = pandas.read_csv(full_name, sep=",")
            df = df.sample(frac=1.0)
            if LIMIT_ROWS is not None and len(df.index) > LIMIT_ROWS:
                df = df.iloc[:LIMIT_ROWS, :]
            print("\n------------ DATASET INFO -----------------")
            print("Data Points in Dataset '%s': %d" % (dataset_file, len(df.index)))
            print("Features in Dataset: " + str(len(df.columns)))

            # Filling NaN and Handling (Removing) constant features
            df = df.fillna(0)
            df = df.loc[:, df.nunique() > 1]
            print("Features in Dataframe after removing constant ones: " + str(len(df.columns)))

            features_no_cat = df.select_dtypes(exclude=['object']).columns
            print("Features in Dataframe after non-numeric ones (including label): " + str(len(features_no_cat)))

            # Initialize LabelEncoder
            le = LabelEncoder()

            # Fit and transform the target variable
            y = le.fit_transform(df[LABEL_NAME].to_numpy())
            unique_y = numpy.unique(y)
            print("Dataset contains %d Classes" % len(unique_y))

            # Set up train test split excluding categorical values that some algorithms cannot handle
            # 1-Hot-Encoding or other approaches may be used instead of removing
            x_no_cat = df.select_dtypes(exclude=['object'])
            x_train, x_test, y_train, y_test = ms.train_test_split(x_no_cat, y, test_size=TT_SPLIT, shuffle=True)

            # Once everyhing is ready, sets up experiments in batches
            print('-------------------- BASE CLASSIFIERS -----------------------')
            exercise_classifier_batch(dataset_name=dataset_file, clf_list=get_base_learners(), tag="BASE",
                                      print_csv=False, x_train=x_train, y_train=y_train,
                                      x_test=x_test, y_test=y_test, unique_classes=unique_y,
                                      existing_exps=existing_exps)

            print('-------------------- DATA DIVERSITY CLASSIFIERS -----------------------')
            exercise_classifier_batch(dataset_name=dataset_file, clf_list=get_data_diversity_learners(), tag="DATA_DIVERSITY",
                                      print_csv=PRINT_CSV, x_train=x_train, y_train=y_train,
                                      x_test=x_test, y_test=y_test, unique_classes=unique_y,
                                      existing_exps=existing_exps)

            print('-------------------- ALG DIVERSITY CLASSIFIERS -----------------------')
            exercise_classifier_batch(dataset_name=dataset_file, clf_list=get_alg_diversity_learners(), tag="ALG_DIVERSITY",
                                      print_csv=PRINT_CSV, x_train=x_train, y_train=y_train,
                                      x_test=x_test, y_test=y_test, unique_classes=unique_y,
                                      existing_exps=existing_exps)

            print('-------------------- DATA AND ALG DIVERSITY CLASSIFIERS -----------------------')
            exercise_classifier_batch(dataset_name=dataset_file, clf_list=get_data_alg_diversity_learners(), tag="DATA_ALG_DIVERSITY",
                                      print_csv=PRINT_CSV, x_train=x_train, y_train=y_train,
                                      x_test=x_test, y_test=y_test, unique_classes=unique_y,
                                      existing_exps=existing_exps)
