"""
For Python 3.4.X
Gradient Boosting Classifier
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.cross_validation import train_test_split
from read_in import get_data

# Parameters
NEST = 100      # Number of estimators
LR = 0.1
LOSS = "deviance"
MDEPTH = 3

# Import Data
X, Y, x_test, test_id = get_data()
x_train, y_train, x_valid, y_valid = train_test_split(X,Y,test_size=0.333)

# Make our model
model = GradientBoostingClassifier(loss=LOSS, n_estimators=NEST, 
                                    learning_rate=LR, max_depth=MDEPTH);
model.fit(x_train,y_train)

# Make metrics on ze data
y_preds = model.predict(x_valid)
cr = classification_report(y_valid,y_preds)
cm = confusion_matrix(y_valid,y_preds)
ac = accuracy_score(y_valid,y_preds)
print(cr); print(cm); print("Accuracy Score: %f" % ac);

# Run classifications on the final test set
y_test = model.predict(x_test);

for i in range(len(y_test)):
    final.append( [ test_id[i],y_test[i] ] )
    
submission = pd.DataFrame(final, columns=["Id","Response"])
submission.to_csv("submission_gbc1.csv",index=False)

"""
Classification Report & Info:
"""
