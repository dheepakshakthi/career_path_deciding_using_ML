import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score

# Load dataset
file_path = "student-scores-6k.csv"
df = pd.read_csv(file_path)

# Encode target variable
label_encoder = LabelEncoder()
df["career_aspiration"] = label_encoder.fit_transform(df["career_aspiration"])

# Define features and target
X = df.drop(columns=["career_aspiration"])
y = df["career_aspiration"]

# Convert boolean columns to int for compatibility
X = X.astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy of the model:", accuracy)


def predict_career(part_time_job, absence_days, extracurricular_activities, weekly_self_study_hours, 
                   math_score, history_score, physics_score, chemistry_score, 
                   biology_score, english_score, geography_score):

    input_data = pd.DataFrame([[part_time_job, absence_days, extracurricular_activities, weekly_self_study_hours,
                                math_score, history_score, physics_score, chemistry_score, 
                                biology_score, english_score, geography_score]],
                              columns=X.columns)
    
    input_data = input_data.astype(int)
    
    predicted_label = model.predict(input_data)[0]
    predicted_career = label_encoder.inverse_transform([predicted_label])[0]
    
    return predicted_career

print("Enter your details to get your best fit career:")
part_time_job_nb = int(input("Do you have a part-time job? (1 for Yes, 0 for No): "))
if part_time_job_nb == 1:
    part_time_job = True
else:
    part_time_job = False
absence_days = int(input("Number of absence days: "))
extracurricular_activities_nb = int(input("Are you involved in extracurricular activities? (1 for Yes, 0 for No): "))
if extracurricular_activities_nb == 1:
    extracurricular_activities = True
else:
    extracurricular_activities = False
weekly_self_study_hours = int(input("Weekly self-study hours: "))
math_score = int(input("Math score: "))
history_score = int(input("History score: "))
physics_score = int(input("Physics score: "))
chemistry_score = int(input("Chemistry score: "))
biology_score = int(input("Biology score: "))
english_score = int(input("English score: "))
geography_score = int(input("Geography score: "))


prediction = predict_career(part_time_job, absence_days, extracurricular_activities, weekly_self_study_hours,
                                math_score, history_score, physics_score, chemistry_score, 
                                biology_score, english_score, geography_score)
print(prediction)

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Data visualization
plt.figure(figsize=(12, 6))
sns.histplot(y_test, color="blue", label="Actual", alpha=0.5, bins=20)
sns.histplot(y_pred, color="red", label="Predicted", alpha=0.5, bins=20)
plt.legend()
plt.title("Distribution of Actual vs Predicted Career Aspirations")

# Feature importance visualization
feature_importances = model.feature_importances_
feature_names = X.columns

#Histogram of feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importance in Career Aspiration Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")

# correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")

#scatter plot for actual vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Career Aspirations")
plt.ylabel("Predicted Career Aspirations")
plt.title("Scatter Plot of Actual vs Predicted Career Aspirations")
plt.show()
