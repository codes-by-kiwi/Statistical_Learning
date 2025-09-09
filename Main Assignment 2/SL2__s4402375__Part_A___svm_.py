
# from google.colab import drive
# drive.mount('/content/drive')

"""#setting file path"""

file_path_1 = '/content/drive/My Drive/SL assignment 2'

"""#converting csv file in drive into df"""

import pandas as pd

file_path_2 = '/content/drive/My Drive/SL assignment 2/MHpredict.csv'
df_MHpredict = pd.read_csv(file_path_2)

"""#inspecting DF"""

print(df_MHpredict)

"""#inspecting columns of DF"""

print(df_MHpredict.columns)

"""#import statements"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

"""#student number defined"""

student_number = 4402375

"""#Y variable set to the dep_sev_fu column"""

Y = df_MHpredict['dep_sev_fu']

"""# X variable set to everything in DF besides dep_sev_fu column"""

X = df_MHpredict.drop(columns=['dep_sev_fu'])

"""#printing the Sexe column in DF"""

print(df_MHpredict['Sexe'])



"""#categorical columns divided into 2
### conversion : Sexe --> Sexe_male and Sexe_female
"""

X = pd.get_dummies(X)

print(X)



"""#splitting dataset"""

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=1000, test_size=500, random_state=student_number
)

"""#inspecting dimensions of train and test portions"""

print("Dimensions of X_train:", X_train.shape)
print("Dimensions of X_test:", X_test.shape)
print("Dimensions of y_train:", Y_train.shape)
print("Dimensions of y_test:", Y_test.shape)

"""#checking numeric and categorical columns"""

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categoric_features = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numerical Features:", numeric_features)
print("Categorical Features:", categoric_features)

print(X_train)



"""#preprocessor transformers defined"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categoric_features)
    ]
)

"""#quick check of preprocessing steps"""

from sklearn.feature_extraction import FeatureHasher
import pandas as pd

X_train_preprocessed = preprocessor.fit_transform(X_train)
print("Transformed shape:", X_train_preprocessed.shape)

one_hot_encoded_features = preprocessor.named_transformers_['cat']
standard_scaled_features = preprocessor.named_transformers_['num']

ohe_feature_names = one_hot_encoded_features.get_feature_names_out(categoric_features)
standard_scaled_features_names = standard_scaled_features.get_feature_names_out(numeric_features)

all_feature_ohe_ss = list(standard_scaled_features_names) + list(ohe_feature_names)

X_train_df = pd.DataFrame(X_train_preprocessed.toarray() if hasattr(X_train_preprocessed, "toarray") else X_train_preprocessed,
                          columns=all_feature_ohe_ss)

print(X_train_df.head())

"""#SVM pipeline creation"""

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svm', SVC())
])

"""#hyperparameter tuning grid defined"""

param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 15, 20, 25, 30],
    'svm__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'svm__kernel': ['rbf', 'poly', 'sigmoid']
}

print(Y_train.value_counts())



"""#Stratified K Fold with number of splits = 10 and original data used"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=student_number)

grid_search = GridSearchCV(svm_pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, Y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Best parameters:", grid_search.best_params_)
print("\nClassification report:\n", classification_report(Y_test, y_pred))

"""#changing param grid"""

param_grid2 = {
    'svm__C': [0.001, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 15, 20, 25, 30],
    'svm__gamma': [0.0001, 0.001, 0.01, 0.1,0.15,0.20,0.30, 1, 10],
    'svm__kernel': ['rbf']
}

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

cv2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=student_number)

grid_search2 = GridSearchCV(svm_pipeline, param_grid2, cv=cv2, scoring='accuracy', n_jobs=-1)
grid_search2.fit(X_train, Y_train)

best_model2 = grid_search2.best_estimator_
y_pred2 = best_model2.predict(X_test)
print("Best parameters:", grid_search2.best_params_)
print("\nClassification report:\n", classification_report(Y_test, y_pred2))

"""#Stratified K Fold with number of splits = 10 and filtered data used"""

train_data = pd.concat([X_train, Y_train], axis=1)
label_counts = Y_train.value_counts()
valid_labels = label_counts[label_counts > 5].index
filtered_train_data = train_data[train_data[Y_train.name].isin(valid_labels)]

X_train_filtered = filtered_train_data.drop(columns=[Y_train.name])
Y_train_filtered = filtered_train_data[Y_train.name]

common_classes = set(Y_train_filtered.unique()) & set(Y_test.unique())
X_test_filtered = X_test[Y_test.isin(common_classes)]
Y_test_filtered = Y_test[Y_test.isin(common_classes)]

cv3 = StratifiedKFold(n_splits=10, shuffle=True, random_state=student_number)
grid_search3 = GridSearchCV(svm_pipeline, param_grid2, cv=cv3, scoring='accuracy', n_jobs=-1)
grid_search3.fit(X_train_filtered, Y_train_filtered)

best_model3 = grid_search3.best_estimator_
y_pred3 = best_model3.predict(X_test_filtered)
print("Best parameters:", grid_search3.best_params_)
print("\nClassification report:\n", classification_report(Y_test_filtered, y_pred3))

"""#changing hyperparameter grid"""

from sklearn.model_selection import RandomizedSearchCV

param_grid3 = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1],
    'svm__kernel': ['rbf', 'linear']
}

train_data = pd.concat([X_train, Y_train], axis=1)
label_counts = Y_train.value_counts()
valid_labels = label_counts[label_counts > 5].index
filtered_train_data = train_data[train_data[Y_train.name].isin(valid_labels)]

X_train_filtered = filtered_train_data.drop(columns=[Y_train.name])
Y_train_filtered = filtered_train_data[Y_train.name]

common_classes = set(Y_train_filtered.unique()) & set(Y_test.unique())
X_test_filtered = X_test[Y_test.isin(common_classes)]
Y_test_filtered = Y_test[Y_test.isin(common_classes)]

cv4 = StratifiedKFold(n_splits=10, shuffle=True, random_state=student_number)

randomized_search1 = RandomizedSearchCV(svm_pipeline, param_grid3, n_iter=10, cv=cv4, scoring='accuracy', n_jobs=-1)
randomized_search1.fit(X_train_filtered, Y_train_filtered)

best_model4 = grid_search.best_estimator_
y_pred4 = best_model4.predict(X_test_filtered)
print("Best parameters:", randomized_search1.best_params_)
print("\nClassification report:\n", classification_report(Y_test_filtered, y_pred4))

"""#applying SMOTE"""

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

smote_creation = SMOTE(random_state=student_number)

train_data = pd.concat([X_train, Y_train], axis=1)
label_counts = Y_train.value_counts()
valid_labels = label_counts[label_counts > 5].index
filtered_train_data = train_data[train_data[Y_train.name].isin(valid_labels)]

X_train_filtered = filtered_train_data.drop(columns=[Y_train.name])
Y_train_filtered = filtered_train_data[Y_train.name]

common_classes = set(Y_train_filtered.unique()) & set(Y_test.unique())
X_test_filtered = X_test[Y_test.isin(common_classes)]
Y_test_filtered = Y_test[Y_test.isin(common_classes)]

X_train_resampled, Y_train_resampled = smote_creation.fit_resample(X_train_filtered, Y_train_filtered)

cv5 = StratifiedKFold(n_splits=10, shuffle=True, random_state=student_number)
# grid_search5 = GridSearchCV(svm_pipeline, param_grid3, cv=cv5, scoring='accuracy', n_jobs=-1)
# grid_search5.fit(X_train_resampled, Y_train_resampled)

randomized_search2 = RandomizedSearchCV(svm_pipeline, param_grid3, n_iter=10, cv=cv5, scoring='accuracy', n_jobs=-1)
randomized_search2.fit(X_train_resampled, Y_train_resampled)

best_model5 = randomized_search2.best_estimator_

y_pred5 = best_model5.predict(X_test_filtered)

print("Best parameters:", randomized_search2.best_params_)
print("\nClassification report:\n", classification_report(Y_test_filtered, y_pred5))

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

preprocessor = best_model5.named_steps['preprocessor']
svm_model = best_model5.named_steps['svm']

X_test_transformed = preprocessor.transform(X_test)

ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categoric_features)
scaled_feature_names = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features)
all_feature_names = list(scaled_feature_names) + list(ohe_feature_names)

result = permutation_importance(
    svm_model,
    X_test_transformed,
    Y_test,
    n_repeats=30,
    random_state=student_number,
    n_jobs=-1
)

feature_importance = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance Mean': result.importances_mean,
    'Importance STD': result.importances_std
})

feature_importance_sorted = feature_importance.sort_values(by='Importance Mean', ascending=False)

print(feature_importance_sorted.head(20))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_sorted['Feature'][:20][::-1],
         feature_importance_sorted['Importance Mean'][:20][::-1])
plt.xlabel('Mean Importance')
plt.title('Top 20 Important Features (Permutation Importance)')
plt.tight_layout()
plt.show()

feature_importance_sorted_negative = feature_importance.sort_values(by='Importance Mean', ascending=True)

print("\nBottom 20 features (Negative Importance):")
print(feature_importance_sorted_negative.head(20))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_sorted_negative['Feature'][:20][::-1],
         feature_importance_sorted_negative['Importance Mean'][:20][::-1])
plt.xlabel('Mean Importance')
plt.title('Bottom 20 Features (Least or Negatively Important)')
plt.tight_layout()
plt.show()

from sklearn.inspection import PartialDependenceDisplay

features_to_plot = ['IDS', 'Age']
PartialDependenceDisplay.from_estimator(
    best_model5,
    X,
    features=features_to_plot,
    target=2,
    kind="average",
    n_jobs=-1,
    grid_resolution=100
)

plt.tight_layout()
plt.show()

from sklearn.inspection import PartialDependenceDisplay

features_to_plot = ['IDS', 'Age']
PartialDependenceDisplay.from_estimator(
    best_model5,
    X,
    features=features_to_plot,
    target=20,
    kind="average",
    n_jobs=-1,
    grid_resolution=100
)

plt.tight_layout()
plt.show()

print(X.columns)

from sklearn.inspection import PartialDependenceDisplay

features_to_plot = ['FQ', 'bSocPhob_Negative']
PartialDependenceDisplay.from_estimator(
    best_model5,
    X,
    features=features_to_plot,
    target=2,
    kind="average",
    n_jobs=-1,
    grid_resolution=100
)

plt.tight_layout()
plt.show()

from sklearn.inspection import PartialDependenceDisplay

features_to_plot = ['FQ', 'bSocPhob_Negative']
PartialDependenceDisplay.from_estimator(
    best_model5,
    X,
    features=features_to_plot,
    target=20,
    kind="average",
    n_jobs=-1,
    grid_resolution=100
)

plt.tight_layout()
plt.show()

negative_true = feature_importance[feature_importance['Feature'] == 'bSocPhob_Negative_True']
negative_false = feature_importance[feature_importance['Feature'] == 'bSocPhob_Negative_False']

plt.figure(figsize=(10, 6))
plt.barh(negative_true['Feature'], negative_true['Importance Mean'], color='blue')
plt.xlabel('Mean Importance')
plt.title('Importance of bSocPhob_Negative_True Feature')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(negative_false['Feature'], negative_false['Importance Mean'], color='red')
plt.xlabel('Mean Importance')
plt.title('Importance of bSocPhob_Negative_False Feature')
plt.tight_layout()
plt.show()

"""#predicting test set"""

from sklearn.metrics import classification_report, accuracy_score

y_pred = best_model5.predict(X_test)

print("Classification Report:\n", classification_report(Y_test, y_pred))

accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

"""David's Diagnosis"""

import pandas as pd

pat_dat = pd.read_csv("Patient.csv")
print(pat_dat.info())
print(pat_dat.head())

print(df_MHpredict.info())

from sklearn.linear_model import LinearRegression

X = df_MHpredict.drop(columns=['dep_sev_fu'])
y = df_MHpredict['dep_sev_fu']

X = pd.get_dummies(X, drop_first=True)
pat_dat = pat_dat.drop(columns=['dep_sev_fu'], errors='ignore')
pat_dat = pd.get_dummies(pat_dat, drop_first=True)
pat_dat = pat_dat.reindex(columns=X.columns, fill_value=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pat_data_scaled = scaler.transform(pat_dat)
model = LinearRegression()
model.fit(X_scaled, y)

predicted_severity = model.predict(pat_data_scaled)[0]
print(f"Predicted severity for David: {predicted_severity}")

if predicted_severity >= 17:
    print("David should be going to the intensive treatment program.")
else:
    print("David should not be going to the intensive treatment program.")