
class NLPSalesAIService:
    def __init__(self,sales_predictor,llm_service, nlp_llm_service, validator):
        self.sales_preditor = sales_predictor
        self.llm_service = llm_service
        self.nlp_llm_service = nlp_llm_service
        self.validator = validator

    def predict_sales(self, user_request):
        print("User Request : ", user_request)
        input_data = self.nlp_llm_service.extract_sales_inputs(user_request)
        print("Input Data : ", input_data)
        self.validator.validate(input_data)
        prediction = self.sales_preditor.predict(input_data)
        explanation = self.llm_service.explain_sales_prediction(prediction,input_data)
        return {
            "predicted_sales": prediction,
            "explanation": explanation
        }




