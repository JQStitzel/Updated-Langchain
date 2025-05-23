import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv("NATL_AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("NATL_AZURE_OPENAI_MODEL_NAME")
deployment = os.getenv("NATL_AZURE_OPENAI_MODEL__DEPLOYMENT_NAME")

subscription_key = os.getenv("NATL_AZURE_OPENAI_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


user_prompt = ""

while (user_prompt != 'stop'):

    user_prompt = input("Enter your question (type 'stop' to exit): ")

    if (user_prompt != 'stop'):        

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            max_completion_tokens=800,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment
        )

        print(response.choices[0].message.content)
    