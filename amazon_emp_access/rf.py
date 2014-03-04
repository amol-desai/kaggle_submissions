from numpy import array, hstack
from sklearn import metrics, cross_validation, ensemble
from scipy import sparse
from itertools import combinations

import numpy as np
import pandas as pd

SEED = 25

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

# This loop essentially from Paul's starter code
def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        #print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N
    
def main(train='train.csv', test='test.csv', submit='logistic_pred.csv'):    
    print "Reading dataset..."
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))

    num_train = np.shape(train_data)[0]
    
    # Transform data
    print "Performing greedy feature selection..."
    y = array(train_data.ACTION)
    good_features = set([])
    Xl = np.zeros((all_data.shape[0],1))
    N = 10
    maxscore = 0
    for deg in range(1,9):
        dp = group_data(all_data, degree=deg) 

        num_features = dp.shape[1]
        
        model = ensemble.RandomForestClassifier(n_estimators=100,n_jobs=-1)
        
        # Xts holds one hot encodings for each individual feature in memory
        # speeding up feature selection 
        #Xts = [OneHotEncoder(dp[:,[i]])[0] for i in range(num_features)]
        Xts = [dp[:,[i]] for i in range(num_features)]
        # Greedy feature selection loop
        for f in range(len(Xts)):
            Xt = np.hstack((Xl,Xts[f]))
            score = cv_loop(Xt[:num_train,1:], y, model, N)
            print "Degree: %i, Feature: %i Mean AUC: %f" % (deg, f, score)
            if (score - maxscore) <= 0.001:
                print "Not taking feature"
            else:
                print "Including feature"
                good_features.add((deg,f))
                Xl = Xt
                maxscore=score
        
        # Remove last added feature from good_features
        print "Selected features %s" % good_features
     
    print "Performing One Hot Encoding on entire dataset..."
    X_train = Xl[:num_train,1:]
    X_test = Xl[num_train:,1:]
    
    print "Training full model..."
    model.fit(X_train, y)
##    from matplotlib import pyplot as plt
##    plt.scatter(range(len(y)),model.predict_proba(X_test)[:,1],c=y,marker='+')
##    plt.show()
    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    create_test_submission(submit, preds)
    
if __name__ == "__main__":
    args = { 'train':  'train.csv',
             'test':   'test.csv',
             'submit': 'rf_pred2.csv' }
    main(**args)  
