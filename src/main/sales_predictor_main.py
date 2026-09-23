from src.services.application_service import SalesAIService
from src.services.llm_service import LLMService
from src.services.sales_predictor_service import SalesPredictor

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

llm_service = LLMService()

sales_ai = SalesAIService(
    predictor,
    llm_service
)
result = sales_ai.predict_sales(tomorrow)

print(result["predicted_sales"])

print(result["explanation"])