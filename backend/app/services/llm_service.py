"""
CARE-AD+ LLM Service

Integration with Ollama for local LLM-powered explanations.
Provides both technical and patient-friendly explanations.
Enhanced with RAG (Retrieval-Augmented Generation) for medical knowledge.
"""
import os
from typing import Dict, Any, Optional
import asyncio

from app.config import settings
from app.services.rag_service import get_rag_service


class LLMService:
    """
    LLM service using Ollama for AI-powered explanations.
    """
    
    def __init__(self):
        self.host = settings.OLLAMA_HOST
        self.model = settings.OLLAMA_MODEL
        self.available = False
        self.rag_service = get_rag_service()  # RAG for knowledge enhancement
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama is available."""
        try:
            import httpx
            response = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            self.available = response.status_code == 200
            if self.available:
                print(f"✅ LLM Service connected to Ollama ({self.model})")
            else:
                print(f"⚠️ Ollama responded but may not be ready")
        except Exception as e:
            print(f"⚠️ LLM Service: Ollama not available at {self.host}")
            print(f"   Run: ollama serve && ollama pull {self.model}")
            self.available = False
    
    def check_availability(self) -> Dict[str, Any]:
        """Check current LLM availability."""
        self._check_connection()
        return {
            "available": self.available,
            "host": self.host,
            "model": self.model,
            "message": "Connected" if self.available else "Ollama not running"
        }
    
    async def generate_response(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "technical"
    ) -> str:
        """
        Generate LLM response (async).
        """
        if not self.available:
            return self._get_fallback_response(message, context, mode)
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_sync,
                message,
                context,
                mode
            )
            return response
        except Exception as e:
            print(f"LLM error: {e}")
            return self._get_fallback_response(message, context, mode)
    
    def _generate_sync(
        self,
        message: str,
        context: Optional[Dict[str, Any]],
        mode: str
    ) -> str:
        """Synchronous generation."""
        import httpx
        
        # Build prompt
        prompt = self._build_prompt(message, context, mode)
        
        try:
            response = httpx.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                return self._get_fallback_response(message, context, mode)
                
        except Exception as e:
            print(f"Ollama request error: {e}")
            return self._get_fallback_response(message, context, mode)
    
    def _build_prompt(
        self,
        message: str,
        context: Optional[Dict[str, Any]],
        mode: str
    ) -> str:
        """Build the prompt for the LLM with RAG enhancement."""
        
        # Get predicted class for RAG
        predicted_class = context.get("predicted_class", "") if context else ""
        
        # Use RAG to enhance prompt with medical knowledge
        if predicted_class and self.rag_service:
            try:
                enhanced_prompt = self.rag_service.enhance_prompt(
                    message, 
                    predicted_class, 
                    mode
                )
                return enhanced_prompt
            except Exception as e:
                print(f"RAG enhancement failed, using basic prompt: {e}")
        
        # Fallback to basic prompt if RAG fails
        if mode == "patient":
            system = """You are a caring medical assistant explaining Alzheimer's disease test results to patients and families. 
Use simple, compassionate language. Avoid medical jargon. Be reassuring but honest."""
        else:
            system = """You are a clinical decision support AI for neurologists analyzing Alzheimer's disease predictions.
Provide technical, evidence-based explanations referencing relevant biomarkers and staging criteria."""
        
        prompt = f"{system}\n\n"
        
        if context:
            prompt += "Context:\n"
            if "predicted_class" in context:
                prompt += f"- Prediction: {context['predicted_class']}\n"
            if "confidence_score" in context:
                prompt += f"- Confidence: {context['confidence_score']*100:.1f}%\n"
            if "patient_name" in context:
                prompt += f"- Patient: {context['patient_name']}\n"
            if "patient_age" in context:
                prompt += f"- Age: {context['patient_age']}\n"
            prompt += "\n"
        
        prompt += f"User: {message}\n\nAssistant:"
        
        return prompt
    
    def _get_fallback_response(
        self,
        message: str,
        context: Optional[Dict[str, Any]],
        mode: str
    ) -> str:
        """Get fallback response when LLM is unavailable."""
        
        predicted_class = context.get("predicted_class", "") if context else ""
        confidence = context.get("confidence_score", 0) if context else 0
        
        if mode == "patient":
            responses = {
                "NonDemented": f"Great news! Based on the brain scan analysis, the results appear normal. The AI system found no significant signs of dementia or Alzheimer's disease. Continue maintaining a healthy lifestyle with regular exercise, good nutrition, and mental stimulation.",
                
                "VeryMildDemented": f"The analysis shows some very early changes that warrant attention. These findings suggest very mild cognitive changes. I recommend scheduling a follow-up appointment with your doctor to discuss these results and plan for monitoring. Early awareness allows for the best possible care planning.",
                
                "MildDemented": f"The brain scan analysis indicates some changes consistent with mild cognitive impairment. Please schedule an appointment with a neurologist to discuss these findings in detail. There are many support options and treatments available. Having early information helps with planning and care.",
                
                "ModerateDemented": f"The analysis shows significant changes in the brain scan. It's important to consult with your healthcare team soon to discuss these results and create a comprehensive care plan. Remember, you're not alone - there are many resources and support systems available."
            }
        else:
            responses = {
                "NonDemented": f"Classification: Cognitively Normal (Confidence: {confidence*100:.1f}%)\n\nThe MRI analysis indicates no significant structural abnormalities associated with Alzheimer's disease. Brain parenchyma appears within normal limits for age. No evidence of hippocampal atrophy or ventricular enlargement. Recommend routine follow-up per standard protocols.",
                
                "VeryMildDemented": f"Classification: Very Mild Cognitive Impairment (Confidence: {confidence*100:.1f}%)\n\nFindings suggestive of CDR stage 0.5. Early hippocampal volume changes detected. Recommend: comprehensive neuropsychological evaluation, CSF biomarker analysis consideration, 6-month follow-up imaging, patient counseling regarding findings.",
                
                "MildDemented": f"Classification: Mild Dementia (Confidence: {confidence*100:.1f}%)\n\nPattern consistent with mild Alzheimer's disease (CDR stage 1). Notable findings: medial temporal lobe atrophy, hippocampal volume reduction, mild ventricular enlargement. Recommend: neurology referral, pharmacological intervention evaluation (ChEIs), functional capacity assessment, care planning initiation.",
                
                "ModerateDemented": f"Classification: Moderate Dementia (Confidence: {confidence*100:.1f}%)\n\nFindings consistent with moderate-stage Alzheimer's disease (CDR stage 2). Significant structural changes including: marked hippocampal atrophy, generalized cortical atrophy, ventricular dilation. Priority: comprehensive care plan, safety assessment, caregiver support, medication review, consideration of supervised living arrangements."
            }
        
        return responses.get(predicted_class, "Analysis complete. Please consult with your healthcare provider for detailed interpretation of these results.")


# Singleton
_llm_service = None

def get_llm_service() -> LLMService:
    """Get or create LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
