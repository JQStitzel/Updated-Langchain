from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from datetime import date
from dotenv import load_dotenv
import os
import json
import pyodbc
import sys

load_dotenv()

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.database import DatabaseConfig
from config.queries import InsuranceQueries

## SET UP THE AZURE OPENAI CONNECTION

endpoint = os.getenv("NATL_AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("NATL_AZURE_OPENAI_MODEL_NAME")
deployment = os.getenv("NATL_AZURE_OPENAI_MODEL__DEPLOYMENT_NAME")

subscription_key = os.getenv("NATL_AZURE_OPENAI_KEY")
api_version = "2024-12-01-preview"

llm = AzureChatOpenAI(
    azure_deployment=deployment,
    openai_api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    temperature=0,    
    model_kwargs={
        "response_format": {"type": "json_object"}  # Place in model_kwargs for clarity
    }
)


# Define a structured prompt
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    Extract the insurance coverage name, the type of data requested (ex: limit, deductible), state, and effective date from the following question.
    If an effective date is not provided, assume """ + str(date.today()) + """.
    If you aren't sure about the coverage name, type of data, or state, ask for clarification. Aside from the default effective date, DO NOT MAKE ASSUMPTIONS ABOUT THEM.
    
    Return your response in valid JSON format with these exact fields:
        "clarification": "yes" or "no",
        "response": 
            "coverage": [coverage name],
            "type": [type of data],
            "state": [state formatted as the two-character state code (ex: California = CA, Ohio = OH)]
            "effective_date": [effective date]
    
    Where 'clarification' is either "yes" or "no" based on whether you need clarification, and 'response' is the text response the user should see.
    
    <question>
    {question}
    </question>
    """
)

chain = {"question": RunnablePassthrough()} | prompt | llm

user_question = ''

while user_question != "stop":

    # User input
    user_question = input("Type to chat ('stop' to quit): ")

    if (user_question != "stop"):

        # Extract structured data
        extracted_data = chain.invoke(user_question)

        # Parse JSON response
        try:
            extracted_data_json = json.loads(extracted_data.content)    
            clarification_needed = extracted_data_json['clarification']

            if clarification_needed == "yes":
                print(extracted_data_json['response'])
            else:
                print("One moment while I get that for you.")
            
            clarification = ""
            
            while clarification_needed == "yes":
                clarification += " | " + input("Type to chat: ")
                
                updated_question = f"Original question: {user_question} | Clarifications: {clarification}"    
                
                extracted_data = chain.invoke(updated_question)
                extracted_data_json = json.loads(extracted_data.content)
                clarification_needed = extracted_data_json['clarification']
                
                if clarification_needed == "yes":
                    print(extracted_data_json['response'])
                else:
                    print("One moment while I get that for you.")
                
        except json.JSONDecodeError:
            print("Error: Response is not valid JSON")
            print("Raw response:", extracted_data.content)



        ## Query system based on data identified

        coverage = extracted_data_json['response']['coverage']
        type = extracted_data_json['response']['type']
        state = extracted_data_json['response']['state']
        effective_date = extracted_data_json['response']['effective_date']


        # Get database configuration
        db_config = DatabaseConfig()
        connection_string = db_config.get_connection_string()
        
        # Connect using the secure connection string
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # Use parameterized query to prevent SQL injection
        query = InsuranceQueries.get_coverage_options_query()
        
        # Execute parameterized query with safe parameters
        cursor.execute(
            query, 
            InsuranceQueries.PRODUCT_IDENTIFIER,
            coverage, 
            coverage,  # Used twice in the query for CoverageCode and CoverageDisplayName
            state, 
            type, 
            effective_date
        )

        # query_template = """
        # *** QUERY WENT HERE ***
        # """

        # sql_query = query_template.format(
        #     coverage=coverage,
        #     state=state,
        #     type=type,
        #     date=effective_date
        # )

        # cursor.execute(sql_query)


        # Fetch all results
        rows = cursor.fetchall()

        answer = ""

        # Print results
        for row in rows:
            answer += f"{row[0]}|"
            # print(row[0])

        conn.close()


        llm = AzureChatOpenAI(
            azure_deployment=deployment,
            openai_api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
            temperature=0
        )

        context = f"""
            coverage: {coverage},
            state: {state},
            type: {type},
            effective date: {effective_date}"""

        # Define a structured prompt
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are a helpful assistant. The user has asked the following question and we have the following answer (pipe delimited). 
            Use only the data provided in your response.
            Respond to the user in a friendly manner and order the data in the answer in a way that would make sense for a person (ex: order values in ascending order).
            If no answer data is provided, respond kindly that no information could be found based on their query and suggest they verify their information and try again.
            Include contextual information for them as well so that they know the criteria that was used in gathering the information.
                
            <question>
            {question}
            </question>

            <context>
            """ + context + """
            </context>
            
            <answer>
            """ + answer + """
            </answer>
            """
        )

        format_chain = {"question": RunnablePassthrough()} | prompt | llm

        # Extract structured data
        response = format_chain.invoke(user_question)

        print(response.content)