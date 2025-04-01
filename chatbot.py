import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from google import genai
from google.genai import types

file_path = "student-scores-6k.csv"
df = pd.read_csv(file_path)
label_encoder = LabelEncoder()
df["career_aspiration"] = label_encoder.fit_transform(df["career_aspiration"])
X = df.drop(columns=["career_aspiration"])
y = df["career_aspiration"]
X = X.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

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

client = genai.Client(api_key="API_KEY")

generate_content_config = types.GenerateContentConfig(
    temperature=1,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
    response_mime_type="text/plain",
    system_instruction=[
        types.Part.from_text(text="""you are an best career path choosing chatbot named \"Apollo\". 
        dont enter their personal space or stories.
        dont give information other than academic career path counselling. if the user asks for career path according to scores, ask for the scores and then use the predict_career function to predict the career path, then provide the answer. if the user asks for general advice, provide general advice."""),
    ],
)

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    if "scores" in user_input.lower() or "score" in user_input.lower() or "marks" in user_input.lower() :
        try:
            part_time_job_nb = int(input("Do you have a part-time job? (1 for Yes, 0 for No): "))
            part_time_job = bool(part_time_job_nb)
            absence_days = int(input("Number of absence days: "))
            extracurricular_activities_nb = int(input("Are you involved in extracurricular activities? (1 for Yes, 0 for No): "))
            extracurricular_activities = bool(extracurricular_activities_nb)
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
            print("Apollo:", prediction)

        except ValueError:
            print("Apollo: Invalid input. Please enter numeric values for scores and 1 or 0 for boolean questions.")

    else:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=user_input, config=generate_content_config)
        print("Apollo:", response.text)