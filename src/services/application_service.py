from src.services.llm_service import LLMService
from src.services.sales_predictor_service import SalesPredictor


class SalesAIService:
    def __init__(self,sales_predictor,llm_service):
        self.sales_preditor = sales_predictor
        self.llm_service = llm_service

    def predict_sales(self, input_data):
        prediction = self.sales_preditor.predict(input_data)
        explanation = self.llm_service.explain_sales_prediction(prediction,input_data)
        return {
            "predicted_sales": prediction,
            "explanation": explanation
        }

