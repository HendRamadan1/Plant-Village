from google import genai
import os
from dotenv import load_dotenv

class GeminiService:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("API_KEY")

        if not api_key:
            raise ValueError("❌ API_KEY not found! Check your .env file.")

        # ✅ إنشاء client
        self.client = genai.Client(api_key=api_key)

    def get_plant_report(self, disease_name, confidence):
        """يرسل الطلب لـ Gemini ويرجع النص"""
        prompt = f"""
Acting as a professional plant pathologist, provide a detailed report for:
Disease: {disease_name}
Detection Confidence: {confidence}

Please provide the following sections:
1. Disease Description
2. Immediate Treatment (Organic and Chemical)
3. Prevention

Keep the tone professional and practical for farmers.
"""

        try:
            # ✅ استدعاء Gemini
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            return response.text

        except Exception as e:
            return f"Error generating AI report: {str(e)}"


# إنشاء instance
gemini_service = GeminiService()