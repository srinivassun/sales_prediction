from src.natural_langugage.nlp_app_service import NLPSalesAIService
from src.natural_langugage.nlp_llm_service import NLPLLMService
from src.natural_langugage.validator import SalesInputValidator
from src.services.llm_service import LLMService
from src.services.sales_predictor_service import SalesPredictor



predictor = SalesPredictor(
    "../../model/sales_model.pkl"
)

llm_service = LLMService()
nlp_llm_service = NLPLLMService()
validator = SalesInputValidator()

sales_ai = NLPSalesAIService(
    predictor,
    llm_service,
    nlp_llm_service,
    validator
)

user_request = input(
    "What would you like to know? "
)

try:

    result = sales_ai.predict_sales(user_request)

    print("\nPrediction:")
    print(
        f"₹{result['predicted_sales']:,.2f}"
    )

    print("\nExplanation:")
    print(
        result["explanation"]
    )

except ValueError as e:

    print("\nInput Error:")
    print(e)
