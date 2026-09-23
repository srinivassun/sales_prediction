from dotenv import load_dotenv

from src.services.sales_predictor_service import SalesPredictor

load_dotenv()
# Predict tomorrow sales prediction by calling sales model
tomorrow = {
    "marketing_spend": 10000,
    "customers": 500,
    "orders": 120,
    "holiday": 0,
    "price": 5000
}
predictor = SalesPredictor(
    "../../model/sales_model.pkl"
)
sales = predictor.predict(tomorrow)
print(sales)

#Integration with LLM Starts here

#Test code to check model is called
#response = client.responses.create(
#    model="openai/gpt-oss-120b",
#    input="Explain machine learning in simple business language."
#)
#print(response.output_text)

prompt = f"""
You are a business sales analyst.

Our machine learning model predicted tomorrow's sales
as ₹{sales:,.0f}.

Explain this prediction in simple business language.

Mention:
1. The predicted sales amount.
2. That this is a machine learning forecast.
3. Keep the explanation concise.
"""
llm_response = predictor.call_llm(prompt)

print(llm_response)
print("------------------------------------------")

new_prompt = f"""
You are a business sales analyst.

Tomorrow's sales forecast:

Predicted Sales: ₹{sales:,.0f}

Business inputs:

Marketing Spend: ₹{tomorrow["marketing_spend"]:,}
Customers: {tomorrow["customers"]}
Orders: {tomorrow["orders"]}
Holiday: {"Yes" if tomorrow["holiday"] else "No"}
Product Price: ₹{tomorrow["price"]:,}

Explain the forecast in simple business language.

Do not invent additional facts.
Clearly distinguish the model's prediction from assumptions.
"""
llm_response = predictor.call_llm(new_prompt)

print(llm_response)