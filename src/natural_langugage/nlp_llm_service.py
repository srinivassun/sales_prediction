# Converts the natural language user input to JSON
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

class NLPLLMService:

    def __init__(self):
        load_dotenv()
        groq_key = os.getenv("GROQ_API_KEY")

        self.client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def extract_sales_inputs(self, user_request):

        prompt = f"""
You are a sales prediction input extraction system.

Extract these fields from the user's request:

- marketing_spend
- customers
- orders
- holiday
- price

Rules:

1. Return ONLY valid JSON.
2. Use numbers for numeric fields.
3. holiday must be 0 or 1.
4. Use null when a value is not provided.
5. Do not invent missing values.

User request:

{user_request}
"""

        response = self.client.responses.create(
            model="openai/gpt-oss-120b",
            input=prompt
        )
        result = response.output_text
        return json.loads(result)