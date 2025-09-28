from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 



def plot_result_table(table):
    """
    Plot classifier accuracies from a result table with
    classifiers on X-axis and accuracy on Y-axis.
    
    Parameters:
        table (pd.DataFrame): DataFrame with feature selection methods as rows
                              and classifiers as columns.
    """
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

def selectkbest(indep_X, dep_Y, n):
    # Step 1: Initialize SelectKBest with Chi-Square test
    test = SelectKBest(score_func=chi2, k=n)
    
    # Step 2: Fit the selector to the features (X) and target (y)
    fit1 = test.fit(indep_X, dep_Y)
    
    # Step 3: Transform the dataset to keep only the top 'n' features
    selectk_features = fit1.transform(indep_X)
    
    # --- Print debug information ---
    print("\n=== SelectKBest Feature Selection ===")
    print(f"Original feature shape: {indep_X.shape}")      # (rows, total features before)
    print(f"Reduced feature shape: {selectk_features.shape}")  # (rows, n features after)
    
    # Get selected feature names
    selected_mask = test.get_support()
    selected_columns = indep_X.columns[test.get_support()]
    selected_scores = fit1.scores_[selected_mask]

    # ----- Quick visualization -----
    try:
        plt.figure(figsize=(8, 5))
        plt.barh(selected_columns, selected_scores)
        plt.xlabel("Chi² Score")
        plt.ylabel("Feature")
        plt.title(f"Top {n} Features Selected by Chi² Test")
        plt.gca().invert_yaxis()  # highest score at the top
        plt.tight_layout()
        plt.show()
    except Exception as e:
        # In case plotting is unavailable (e.g., headless env), don't break execution
        print(f"(Plot skipped: {e})")
    
    # Step 4: Return the reduced features (NumPy array)
    return selectk_features

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