import pandas as pd
from sklearn.model_selection import train_test_split 
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def plot_result_table(table):
    # Take the first row (ChiSquare here)
    method = table.index[0]
    values = table.iloc[0].values
    classifiers = table.columns.tolist()

    plt.figure(figsize=(8,5))
    bars = plt.bar(classifiers, values, color="skyblue")

    # Add values on top of bars
    for bar, acc in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{acc:.2f}", ha='center', va='bottom', fontsize=10)

    plt.ylabel(f"Accuracy ({method})", fontsize=12)
    plt.xlabel("Classifiers", fontsize=12)
    plt.title(f"Classifier Performance using {method} Feature Selection", fontsize=14, fontweight="bold")
    plt.ylim(0.8, 1.0)  # focus on high accuracy range
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
def prepare_from_df(df, target="Loan_Status_Y"):
    # Target to numeric
    y, _ = pd.factorize(df[target])

    # Features are everything else (already encoded & imputed)
    X = df.drop(columns=[target])

    # Scale to [0,1] for chi2
    X_scaled = MinMaxScaler().fit_transform(X)

    return X, X_scaled, y
    
def selectk_Classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf): 
    
    dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Logistic','SVMl','SVMnl','KNN','Navie','Decision','Random'])
    for number,idex in enumerate(dataframe.index):      
        dataframe['Logistic'][idex]=acclog[number]       
        dataframe['SVMl'][idex]=accsvml[number]
        dataframe['SVMnl'][idex]=accsvmnl[number]
        dataframe['KNN'][idex]=accknn[number]
        dataframe['Navie'][idex]=accnav[number]
        dataframe['Decision'][idex]=accdes[number]
        dataframe['Random'][idex]=accrf[number]
    return dataframe

def random(X_train,y_train,X_test):
        
        # Fitting K-NN to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
        return  classifier,Accuracy,report,X_test,y_test,cm
def Decision(X_train,y_train,X_test):
        
        # Fitting K-NN to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
        return  classifier,Accuracy,report,X_test,y_test,cm     
    
def knn(X_train,y_train,X_test):
           
        # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
        return  classifier,Accuracy,report,X_test,y_test,cm
    
def Navie(X_train,y_train,X_test):       
        # Fitting K-NN to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
        return  classifier,Accuracy,report,X_test,y_test,cm  
    
def svm_NL(X_train,y_train,X_test):
                
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
        return  classifier,Accuracy,report,X_test,y_test,cm

def svm_linear(X_train,y_train,X_test):
                
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
        return  classifier,Accuracy,report,X_test,y_test,cm


def logistic(X_train,y_train,X_test):       
        # Fitting K-NN to the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
        return  classifier,Accuracy,report,X_test,y_test,cm    

def cm_prediction(classifier,X_test):
     y_pred = classifier.predict(X_test)
        
        # Making the Confusion Matrix
     from sklearn.metrics import confusion_matrix
     cm = confusion_matrix(y_test, y_pred)
        
     from sklearn.metrics import accuracy_score 
     from sklearn.metrics import classification_report 
        #from sklearn.metrics import confusion_matrix
        #cm = confusion_matrix(y_test, y_pred)
        
     Accuracy=accuracy_score(y_test, y_pred )
        
     report=classification_report(y_test, y_pred)
     return  classifier,Accuracy,report,X_test,y_test,cm

def split_scalar(indep_X,dep_Y):
        # Step 1: Split into training (75%) and testing (25%) sets
        X_train, X_test, y_train, y_test = train_test_split(
            indep_X, dep_Y, test_size=0.25, random_state=0
        )
    
        # Step 2: Initialize the StandardScaler
        sc = StandardScaler()
    
        # Step 3: Fit scaler on training data and transform it
        X_train = sc.fit_transform(X_train)
    
        # Step 4: Transform test data using same scaler (to avoid data leakage)
        X_test = sc.transform(X_test)
    
        # Step 5: Return scaled features and labels
        return X_train, X_test, y_train, y_test

def selectkbest(X: pd.DataFrame, y, k: int):
    X = X.copy()
    y = np.asarray(y)

    # --- split columns ---
    cat_cols, num_cols = [], []
    for c in X.columns:
        s = X[c]
        if s.dtype == bool:
            cat_cols.append(c)
        elif np.issubdtype(s.dtype, np.integer):
            if s.nunique(dropna=True) <= 10 or set(s.dropna().unique()).issubset({0,1}):
                cat_cols.append(c)
            else:
                num_cols.append(c)
        elif np.issubdtype(s.dtype, np.floating):
            num_cols.append(c)

    rows = []

    # --- chi2 for categorical (non-negative) ---
    if cat_cols:
        X_cat = X[cat_cols].fillna(0)
        if (X_cat.values < 0).any():
            X_cat[:] = MinMaxScaler().fit_transform(X_cat)
        chi_s, chi_p = chi2(X_cat, y)
        rows += [{"Feature": c, "Method": "chi2", "Score": s, "P": p}
                 for c, s, p in zip(cat_cols, chi_s, chi_p)]

    # --- ANOVA F for numeric ---
    if num_cols:
        X_num = X[num_cols].apply(lambda col: col.fillna(col.median()))
        f_s, f_p = f_classif(X_num, y)
        rows += [{"Feature": c, "Method": "f_classif", "Score": float(s), "P": float(p)}
                 for c, s, p in zip(num_cols, f_s, f_p)]

    if not rows:
        raise ValueError("No features to score.")

    scores = pd.DataFrame(rows)
    scores["Rank"] = -np.log10(np.where(scores["P"] == 0, np.nextafter(0, 1), scores["P"]))
    scores = scores.sort_values(["Rank", "Score"], ascending=False, ignore_index=True)

    selected = scores["Feature"].head(k)
    X_selected = X[selected].to_numpy()

    # quick glance
    print(scores.head(k)[["Feature", "Method", "Score", "P"]])
    return X_selected, selected, scores
