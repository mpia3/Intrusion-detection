import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
# import numpy as np

PATH_TRAIN_DATASET = './dataset/Train_OneClsNumeric.csv'  # percorso del dataset di train
PATH_TEST_DATASET = './dataset/Test_OneClsNumeric.csv'  # percorso del dataset di test
K_FEATURES = 10  # paramentro che indica il numero di feature da estrarre dal ranking di feature
K_COMPONENTS = 10  # parametro che indica il numero di componenti da considerare per la PCA
K_FOLD = 5  # numero di fold per eseguire la k fold cross validation
LIST_RANDOMIZATION = ["sqrt", "log2"]  # configurazione per il parametro max_features della random forest
LIST_BOOTSTRAP = [0.5, 0.6, 0.7, 0.8, 0.9]  # # configurazione per il parametro max_samples della random forest
LIST_N_ESTIMATORS = [10, 20, 30]  # configurazione per il parametro n_estimators della random forest
K_NEIGHBORS = 3  # parametro che definisce il numero di vicini da prendere in considerazione per effettuare la predizione con il classificatore knn


"Carica il dataset in csv dal path specificato"
def load(filepath):
    dataset = pd.read_csv(filepath)
    return dataset


"esplora i dati del dataset"
def data_exploration(dataset):
    count = 0
    columns = dataset.columns
    for c in columns:
        if c == columns[-1]:
            break
        describe(dataset, c)
        boxplot(dataset, c, columns, count)
        scatterplot(dataset, c, columns, count)
        count += 1
        # print(dataset[c].describe())
    # for c in columns:
        # dataset.boxplot(column=c, by=columns[-1])
        # dataset.plot.scatter(x=c, y=columns[-1])
        # plt.show()


"statistiche come min, max, sdev, media ecc..."
def describe(dataset, c):
    print(dataset[c].describe())
    print(sep='\n\n')


"crea il boxplot per una variabile e lo confronta con la label classification"
def boxplot(dataset, c, columns, n):
    dataset.boxplot(column=c, by=columns[-1])
    plt.savefig('./doc/images/'+str(n)+'_boxplot_'+c+'.png')
    plt.show()


"crea il scatterplot per una variabile e lo confronta con la label classification"
def scatterplot(dataset, c, columns, n):
    dataset.plot.scatter(x=c, y=columns[-1])
    plt.savefig('./doc/images/'+str(n)+'_scatterplot_'+c+'.png')
    plt.show()


"feature selection sul dataset"
def feature_selection(dataset, k):
    ranking = mutual_info(dataset)
    list_of_top_features = top_list(ranking, k)
    list_of_top_features.append(dataset.columns[-1])
    return new_data_frame(dataset, list_of_top_features)


"la mutual info (MI) tra due variabili casuali è un valore non negativo," \
"che misura la dipendenza tra le variabili." \
"È uguale a zero se e solo se due variabili casuali sono indipendenti," \
"e valori più alti significano dipendenza più alta." \
"La funzione si basa su metodi non parametrici basati sulla stima dell'entropia dalle distanze" \
"dei k-nearest neighbors"
def mutual_info(dataset):
    columns = list(dataset.columns)
    columns.pop(len(columns)-1)
    # columns = tuple(columns)
    # dataset = dataset.to_numpy()
    # x = dataset[:, :-1]
    # target = dataset[:, -1]
    x, target = split_into_x_y(dataset)
    feature = dict(zip(columns, mutual_info_classif(X=x, y=target)))  # non ordinate per importanza
    sorted_feature = sorted(feature.items(), key=lambda kv: kv[1], reverse=True)
    # print("END feature selection")
    return sorted_feature


"Estrae le prime k feature"
def top_list(list_of_items, k):
    new_list = list()
    j = 1
    for i in list_of_items:
        if j > k:
            break
        new_list.append(i[0])
        j += 1
    return new_list


"Ridimensiona il dataset rispetto ad una lista di feature"
def new_data_frame(dataset, list_of_features):
    return dataset.loc[:, list_of_features]


# def data_transformation(dataset):
    # None


"crea un oggetto PCA fittato sul dataset"
def principal_component_analysis(x, k):
    # x, y = separate_into_x_y(dataset)
    pca = PCA(n_components=k)
    pca.fit(X=x)
    print("Percentage of variance explained: "+str(sum(pca.explained_variance_ratio_)))
    return pca


"esegue la trasformazione del dataset con un oggetto pca"
def apply_pca_to_dataset(x, y, pca):
    new_dataset = pd.DataFrame(pca.transform(x))
    new_dataset.insert(loc=len(new_dataset.columns), column=len(new_dataset.columns + 1), value=y)
    return new_dataset


"applica il k fold cross validation stratificato"
def stratified_k_fold_validation(x, y, n_fold):
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=None)
    x_train = []; x_test = []; y_train = []; y_test = [];
    for train_index, test_index in skf.split(X=x, y=y):
        x_train.append(x.iloc[train_index])
        x_test.append(x.iloc[test_index])
        y_train.append(y.iloc[train_index])
        y_test.append(y.iloc[test_index])

    return x_train, x_test, y_train, y_test


"costruisce una Random Forest in accordo ai parametri passati. Restituisce la Random Forest costruita" \
"n_estimators il numero di alberi nella foresta" \
"min_samples_split il numero minimo di campioni richiesti per dividere un nodo interno" \
"Se float, allora min_samples_split è una frazione e " \
"ceil (min_samples_split * n_samples) è il numero minimo di campioni per ogni divisione." \
"max_features il numero di features da considerare quando si cerca la migliore suddivisione:" \
"Se sqrt, allora max_features = sqrt (n_features), Se log2, allora max_features = log2 (n_features)" \
"max_samples se bootstrap è True, il numero di campioni da estrarre da X per addestrare ogni stimatore di base." \
"Se è None (predefinito), disegnare campioni X.shape [0]." \
"Se int, disegna campioni max_samples." \
"Se float, disegna max_samples * X.shape [0] samples. Pertanto, max_samples dovrebbe essere nell'intervallo (0, 1)."
def random_forest_configuration(X, y, randomization, bootstrap, n_estimators):
    clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=0.05,
                                 max_features=randomization, max_samples=bootstrap)
    clf.fit(X, y)

    return clf


"usa clf per predire le etichette di X. Calcola le metriche accuracy_score, " \
"balanced_accuracy_score, precision_score, recall_score ed f1_score usando i valori predetti " \
"e i valori reali. Restituisce la lista delle metriche calcolate"
def evaluate(X, y_true, clf):
    metrics = []
    y_pred = clf.predict(X)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import balanced_accuracy_score
    metrics.append(accuracy_score(y_true, y_pred))  # oa
    metrics.append(balanced_accuracy_score(y_true, y_pred))  # balanced accuracy
    metrics.append(precision_score(y_true, y_pred))  # precision
    metrics.append(recall_score(y_true, y_pred))  # recall
    metrics.append(f1_score(y_true, y_pred))  # fscore
    return metrics


"valuta la configurazione n_estimators,randomization,bootstrap nell’apprendimento " \
"della Random Forest usando la metrica fscore."
def evaluateCV(kfolds, ListXTrain, ListXTest, ListYTrain, ListYTest, n_estimators, randomization, bootstrap):
    avgTrain = [0.0, 0.0, 0.0, 0.0, 0.0]
    avgTest = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(kfolds):
        clf = random_forest_configuration(ListXTrain[i], ListYTrain[i], randomization, bootstrap, n_estimators)
        metrics = evaluate(ListXTrain[i], ListYTrain[i], clf)
        avgTrain = [x + y for x, y in zip(avgTrain, metrics)]
        metrics = evaluate(ListXTest[i], ListYTest[i], clf)
        avgTest = [x + y for x, y in zip(avgTest, metrics)]

    avgTrain = [x / kfolds for x in avgTrain]
    avgTest = [x / kfolds for x in avgTest]

    return avgTrain, avgTest


"Valuta tutte le configurazioni disponibili per list_randomization, list_bootstrap, list_n_esimators." \
"Restituisce la miglior configurazione randomization, bootstrap , n estimator"
def best_learned_configuration(X_train, X_test, Y_train, Y_test, kfolds, list_randomization, list_bootstrap, list_n_estimators):
    best_f_score = 0.0
    best_randomization = ""
    best_bootstrap = 0.0
    best_n_estimator = 0
    i = 1

    for randomization in list_randomization:
        for bootstrap in list_bootstrap:
            for estimator in list_n_estimators:
                print("configuration n"+str(i)+": " + "randomization="+randomization+
                      " bootstrap="+str(bootstrap)+" n_estimator="+str(estimator))
                avgTrain, avgTest = evaluateCV(kfolds, X_train, X_test, Y_train, Y_test, estimator, randomization, bootstrap)
                if avgTest[-1] > best_f_score:
                    best_f_score = avgTest[-1]
                    best_randomization = randomization
                    best_bootstrap = bootstrap
                    best_n_estimator = estimator
                    print("Best configuration for now: "+best_randomization+" "+str(best_bootstrap)+" "
                          +str(best_n_estimator)+" (fscore="+str(avgTest[-1])+")")
                i += 1

    return {"randomization":best_randomization, "bootstrap":best_bootstrap, "n_estimators":best_n_estimator}  # [best_randomization, best_bootstrap, best_n_estimator]


"applica la PCA a ciascun fold del CV in modo che si possano valutare i punti 11.a e 11.b sugli stessi fold di CV"
def transformation_fold_CV_PCA(kfolds, ListXTrain, ListYTrain, ListXTest, ListYTest):
    new_ListXTrain = []
    new_ListXTest = []

    for i in range(kfolds):
        pca = principal_component_analysis(ListXTrain[i], K_COMPONENTS)
        new_data = apply_pca_to_dataset(ListXTrain[i], ListYTrain[i], pca)
        x, y = split_into_x_y(new_data)
        new_ListXTrain.append(x)
        new_data = apply_pca_to_dataset(ListXTest[i], ListYTest[i], pca)
        x, y = split_into_x_y(new_data)
        new_ListXTest.append(x)

    return new_ListXTrain, new_ListXTest


"Lo stacker della Random Forest al punto 12.a e della Random Forest al punto 12.b. Per lo stacking usare il classificator knn"
def get_stacker(y_pred_model_1, y_pred_model_2, y_true):
    X_train_for_stacking = pd.DataFrame({"Pred_model_1":y_pred_model_1 , "Pred_model_2":y_pred_model_2})
    level1 = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    level1.fit(X_train_for_stacking, y_true)
    return level1


"divide il dataset in due parti x (variabili indipendenti) e y (variabile dipendente)"
def split_into_x_y(dataset):
    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


"stampa una singola lista di score"
def print_evaluation(evaluation):
    print("overall_accuracy_score="+str(evaluation[0])+" balanced_accuracy_score="+str(evaluation[1])+
          " precision_score="+str(evaluation[2])+" recall_score="+str(evaluation[3])+" f1_score="+str(evaluation[4]))
    print(sep='\n\n')


"stampa tutte le liste di score"
def print_all_evaluation(evaluations):
    print("Score su dataset test vergine")
    print_evaluation(evaluations[0])  # pattern su dataset vergine
    print("Score su dataset test con PCA")
    print_evaluation(evaluations[1])  # pattern su dataset con PCA
    print("Score su dataset test per lo stacker")
    print_evaluation(evaluations[2])  # pattern su dateset creato con le predizioni precedenti


if __name__ == '__main__':
    dataset = load(PATH_TRAIN_DATASET)
    print(dataset.shape)  # dimensioni dataset
    print(dataset.head())  # intestazione
    print(dataset.columns)  # colonne
    print(sep='\n\n')

    # data_exploration(dataset)
    # new_data = feature_selection(dataset, K_FEATURES)

    X, y = split_into_x_y(dataset)
    X_train, X_test, Y_train, Y_test = stratified_k_fold_validation(X, y, K_FOLD)

    "Random Forest costruita dal dataset originale variando randomization tra sqrt e log2, " \
    "n_estimators tra 10, 20 e 30, bootstrap tra 0.5,0.6,0.7, 0.8 e 0.9"
    "Random Forest costruita dalle 10 top componenti principali variando randomization tra sqrt e log2, " \
    "n_estimators tra 10, 20 e 30, bootstrap tra 0.5,0.6,0.7, 0.8 e 0.9"
    dict_configuration_original_dataset = dict()
    dict_configuration_PCA = dict()
    print("Best configuration phase...")
    "primo punto (11.a)"
    dict_configuration_original_dataset = best_learned_configuration(X_train, X_test, Y_train, Y_test,
                                                             K_FOLD, LIST_RANDOMIZATION,
                                                             LIST_BOOTSTRAP, LIST_N_ESTIMATORS)
    print("Best configuration for original dataset: randomization="+str(dict_configuration_original_dataset["randomization"])+
          " bootstrap="+str(dict_configuration_original_dataset["bootstrap"])+
          " n_estimators="+str(dict_configuration_original_dataset["n_estimators"]))
    print(sep='\n\n')
    "secondo punto (11.b)"
    new_X_train, new_X_test = transformation_fold_CV_PCA(K_FOLD, X_train, Y_train, X_test, Y_test)
    dict_configuration_PCA = best_learned_configuration(new_X_train, new_X_test, Y_train, Y_test,
                                                        K_FOLD, LIST_RANDOMIZATION,
                                                        LIST_BOOTSTRAP, LIST_N_ESTIMATORS)
    print("Best configuration for PCA dataset: randomization=" + str(dict_configuration_PCA["randomization"]) +
          " bootstrap=" + str(dict_configuration_PCA["bootstrap"]) +
          " n_estimators=" + str(dict_configuration_PCA["n_estimators"]))
    print(sep='\n\n')

    "Usando l’intero dataset Train_OneClsNumeric.csv apprendere" \
    "La Random Forest con la configurazione reputata migliore al punto 11.a" \
    "La Random Forest con la configurazione reputate migliore al punto 11.b" \
    "Lo stacker della Random Forest al punto 12.a e della Random Forest al punto 12.b. Per lo stacking usare il classificator knn"
    print("Train phase...")
    random_forest_original_dataset = random_forest_configuration(X, y, dict_configuration_original_dataset["randomization"],
                                                                 dict_configuration_original_dataset["bootstrap"],
                                                                 dict_configuration_original_dataset["n_estimators"])

    pca = principal_component_analysis(X, K_COMPONENTS)
    dataset_PCA = apply_pca_to_dataset(X, y, pca)
    X_PCA, y_PCA = split_into_x_y(dataset_PCA)
    random_forest_PCA = random_forest_configuration(X_PCA, y_PCA, dict_configuration_PCA["randomization"],
                                                    dict_configuration_PCA["bootstrap"],
                                                    dict_configuration_PCA["n_estimators"])
    print("Stacker train phase...")
    y_RF_original_dataset = random_forest_original_dataset.predict(X)
    y_RF_PCA = random_forest_PCA.predict(X_PCA)
    stacker = get_stacker(y_RF_original_dataset, y_RF_PCA, y)
    print(sep='\n\n')

    "Valutare l’accuratezza dei tre pattern di classificazione appresi al punto 12 " \
    "usando i dati in in Test_OneClsNumeric.csv calcolare le diverse metriche di accuratezza per tali dati."
    print("Valutation phase...")
    test_dataset = load(PATH_TEST_DATASET)
    X_test_dataset, y_test_dataset = split_into_x_y(test_dataset)
    evaluation_RF_original_dataset = evaluate(X_test_dataset, y_test_dataset, random_forest_original_dataset)
    test_dataset_PCA = apply_pca_to_dataset(X_test_dataset, y_test_dataset, pca) #############################################################
    X_test_dataset_PCA, y_test_dataset_PCA = split_into_x_y(test_dataset_PCA)
    evaluation_RF_PCA = evaluate(X_test_dataset_PCA, y_test_dataset_PCA, random_forest_PCA)
    X_test_stacker = pd.DataFrame({"Pred_model_1":random_forest_original_dataset.predict(X_test_dataset),
                                   "Pred_model_2":random_forest_PCA.predict(X_test_dataset_PCA)})
    evaluation_stacker = evaluate(X_test_stacker, y_test_dataset, stacker)

    "confronto con i pattern appresi sul dataset di test"
    print_all_evaluation([evaluation_RF_original_dataset, evaluation_RF_PCA, evaluation_stacker])
