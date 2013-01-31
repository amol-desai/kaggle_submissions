import csv as csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pylab as pl
from sklearn import clone

header = []

#read in csv file and convert to usable np array
def csv2data(csvPath, cols2remove, hasHeader=True):
    global header
    data=[]
    csv_object = csv.reader(open(csvPath,'rb'))
    if hasHeader:
        header = csv_object.next()
    else:
        data.append(csv_object.next())
        header = range(len(data[0]))
    for row in csv_object:
        data.append(row)
    data = np.array(data)
    for col in cols2remove:
        if type(col) is str:
            #cols2remove[cols2remove.index(col)] = header.index(col)
            data = np.delete(data,header.index(col),1)
            header.remove(col)
        else:
            data = np.delete(data,col,1)
            header.pop(col)
    #data = np.delete(data,cols2remove,1)
    return data

#replace each unique string value in specified column with int identifier. Skip blanks.
def replaceStringWithInt(data, cols):
    str2int_map = {}
    for col in cols:
        if type(col) is str:
            col = header.index(col)
        i = 0
        for row in range(len(data)):
            if data[row][col] not in str2int_map and data[row][col] != "":
                str2int_map[data[row][col]] = i
                data[row][col] = i
                i += 1
            elif data[row][col] != "":
                data[row][col] = str2int_map[data[row][col]]
    print str2int_map
    return data

#Converts numbers in data from str to float. If non-numerical data is found, prints row.
def dataStr2Float(data):
    for row in range(len(data)):
        try:
            data[row,:] = data[row,:].astype(np.float)
        except ValueError:
            print str(row)+": "+str(data[row,:])
    return data
    
# % of total at each age
def createHistForIncompleteData(data,col):
    hist = [[0],[0]]
    if type(col) is str:
        col = header.index(col)
    for row in range(len(data)):
        if data[row,col] != '':
            if data[row,col] not in hist[0]:
                hist[0].append(data[row][col].astype(np.float))
                hist[1].append(1)
            else:
                hist[1][hist[0].index(data[row,col])] += 1
    hist = np.array(hist)
    hist[1,0::]=hist[1,0::].astype(np.float)/np.sum(hist[1,0::].astype(np.float))
    return hist

def titanicPrediction():
    data = csv2data("C:/data_sandbox/Titanic/train.csv",['name','ticket','cabin','age'])
    data = replaceStringWithInt(data,['sex','embarked'])
    data[61,6] = 0.0
    data[829,6] = 0.0
    data = dataStr2Float(data)
    Forest = RandomForestClassifier(n_estimators = 100, compute_importances = True, oob_score = True)
    Forest = Forest.fit(data[0::,1::],data[0::,0])
    print "Random Forest OOB Score = ",Forest.oob_score_
    test = csv2data("C:/data_sandbox/Titanic/test.csv",['name','ticket','cabin','age'])
    test = replaceStringWithInt(test,['sex','embarked'])
    test = dataStr2Float(test)
    test[152,4] = 0.0
    Output=Forest.predict(test)
    header.insert(0,"survived")
    plotFeatureImportance(Forest)
    try:
        hist = createHistForIncompleteData(data,"age")
        print hist
    except:
        pass
    writeResults("C:/data_sandbox/Titanic/RF1.csv",Output,"C:/data_sandbox/Titanic/test.csv")

def plotFeatureImportance(forest):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print "Feature ranking:"

    for f in xrange(len(importances)):
        print "%d. feature %s (%f)" % (f + 1, header[indices[f]+1], importances[indices[f]])

    # Plot the feature importances of the forest
    pl.figure()
    pl.title("Feature importances")
    pl.bar(xrange(len(importances)), importances[indices],
           color="r", yerr=std[indices], align="center")
    pl.xticks(xrange(len(importances)), [header[i+1] for i in indices])
    pl.xlim([-1, len(importances)])
    pl.show()

def plotDecisionSurfaces(data,forest):
        # Parameters
    n_classes = forest.n_classes_[0]
    #n_estimators = 30
    plot_colors = "bry"
    plot_step = 0.5

    # Load data
    #iris = load_iris()

    plot_idx = 1

    for f1 in range(forest.n_features_):
        for f2 in range(f1+1,forest.n_features_):
             # We only take the two corresponding features
            X = data[:, [f1+1,f2+1]]
            y = data[:,0]

            # Shuffle
            idx = np.arange(X.shape[0])
            np.random.seed(13)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # Standardize
            mean = X.astype(float).mean(axis=0)
            std = X.astype(float).std(axis=0)
            X = np.divide(np.subtract(X.astype(float),mean),std)
            X = X.astype(float)
            
            # Train
            clf = clone(forest)
            clf = clf.fit(X, y)

            # Plot the decision boundary
            pl.subplot(5, 3, plot_idx)
            
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))

##            if isinstance(model, DecisionTreeClassifier):
##                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
##                Z = Z.reshape(xx.shape)
##                cs = pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
##            else:
            for tree in clf.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = pl.contourf(xx, yy, Z, alpha=0.1, cmap=pl.cm.Paired)

            pl.axis("tight")

            # Plot the training points
            for i, c in zip(xrange(n_classes), plot_colors):
                idx = np.where(y.astype(float) == i)
                pl.scatter(X[idx, 0], X[idx, 1], c=c, label=forest.classes_[0][i],
                        cmap=pl.cm.Paired)

            pl.axis("tight")

            plot_idx += 1

    pl.suptitle("Decision surfaces of a random forest")
    pl.show()

def writeResults(resultsPath,results,testPath,hasHeader=True):
    results_file_object = csv.writer(open(resultsPath, "wb"))
    test_file_object = csv.reader(open(testPath,'rb'))
    if hasHeader:
        results_file_object.writerow(test_file_object.next())
    i = 0
    for row in test_file_object:
        row.insert(0,results[i])
        results_file_object.writerow(row)
        i += 1
    return

