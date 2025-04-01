from google import genai
from google.genai import types

client = genai.Client(api_key="API_KEY")

generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""you are an best career path choosing chatbot named \"Apollo\".  
dont enter their personal space or stories
dont give information other than academic career path counselling"""),
        ],
)

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    response = client.models.generate_content(model="gemini-2.0-flash", contents=user_input, config=generate_content_config)
    print("Apollo:",response.text)
