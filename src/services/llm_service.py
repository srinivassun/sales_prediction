import joblib
import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI

class LLMService:
    def __init__(self):
        load_dotenv()
        groq_key = os.getenv("GROQ_API_KEY")

        self.client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def explain_sales_prediction(self, prediction, input_data) -> str:
        prompt = f"""
        You are a business sales analyst.

        The machine learning model predicted
        tomorrow's sales as ₹{prediction:,.0f}.

        Business inputs:

        Marketing Spend:
        ₹{input_data["marketing_spend"]:,}

        Customers:
        {input_data["customers"]}

        Orders:
        {input_data["orders"]}

        Holiday:
        {"Yes" if input_data["holiday"] else "No"}

        Price:
        ₹{input_data["price"]:,}

        Explain this prediction in simple business language.

        Do not invent facts.
        """
        response = self.client.responses.create(
            model="openai/gpt-oss-120b",
            input=prompt
        )
        return response.output_text