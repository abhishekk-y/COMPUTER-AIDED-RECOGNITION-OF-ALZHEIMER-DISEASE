"""
CARE-AD+ LLM Service

Local LLM integration for generating context-aware explanations.
Uses Ollama for local deployment of models like Llama 3.1 or Mistral.
"""
import os
import json
from typing import Dict, Optional, Any, List
import asyncio

from app.config import settings


class LLMService:
    """
    LLM Service for generating intelligent explanations of predictions.
    
    Supports two modes:
    - technical: Medical terminology for clinicians
    - patient: Simplified language for patients and families
    """
    
    def __init__(self):
        self.ollama_host = settings.OLLAMA_HOST
        self.model_name = settings.OLLAMA_MODEL
        self._client = None
    
    def _get_client(self):
        """Get or create Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.ollama_host)
            except ImportError:
                print("⚠️ Ollama package not installed. Using fallback responses.")
                self._client = None
        return self._client
    
    def _build_system_prompt(self, mode: str) -> str:
        """Build the system prompt based on mode."""
        if mode == "technical":
            return """You are CARE-AD+, an advanced AI assistant specialized in explaining 
Alzheimer's disease prediction results to healthcare professionals.

Your responses should:
- Use appropriate medical terminology
- Reference relevant brain regions (hippocampus, temporal lobe, entorhinal cortex, etc.)
- Explain the clinical significance of findings
- Mention relevant biomarkers and cognitive assessments
- Be precise and evidence-based
- Include differential diagnosis considerations when appropriate

Always maintain a professional, clinical tone while being informative and helpful."""
        
        else:  # patient mode
            return """You are CARE-AD+, a compassionate AI assistant helping explain 
brain health assessment results to patients and their families.

Your responses should:
- Use simple, everyday language (avoid medical jargon)
- Be warm, supportive, and reassuring
- Explain concepts using analogies when helpful
- Acknowledge the emotional aspects of such discussions
- Encourage follow-up with healthcare providers
- Avoid making definitive diagnoses (emphasize this is a screening tool)

Remember: You are explaining results to someone who may be anxious about their 
or their loved one's cognitive health. Be kind, clear, and supportive."""
    
    def _build_context_prompt(self, context: Optional[Dict[str, Any]]) -> str:
        """Build context information for the prompt."""
        if not context:
            return "No specific prediction context available."
        
        parts = []
        
        if "predicted_class" in context:
            class_descriptions = {
                "NonDemented": "Cognitively Normal - No significant signs of dementia detected",
                "VeryMildDemented": "Very Mild Cognitive Impairment - Early changes that may warrant monitoring",
                "MildDemented": "Mild Dementia - Noticeable cognitive changes consistent with early-stage Alzheimer's",
                "ModerateDemented": "Moderate Dementia - Significant cognitive impairment requiring attention"
            }
            desc = class_descriptions.get(context["predicted_class"], context["predicted_class"])
            parts.append(f"Prediction Result: {desc}")
        
        if "confidence_score" in context:
            parts.append(f"Confidence Score: {context['confidence_score']*100:.1f}%")
        
        if "probabilities" in context and context["probabilities"]:
            probs = context["probabilities"]
            parts.append("Class Probabilities:")
            for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                parts.append(f"  - {class_name}: {prob*100:.1f}%")
        
        if "clinical_features" in context and context["clinical_features"]:
            features = context["clinical_features"]
            parts.append("\nClinical Features:")
            if features.get("age"):
                parts.append(f"  - Age: {features['age']} years")
            if features.get("mmse_score"):
                parts.append(f"  - MMSE Score: {features['mmse_score']}/30")
            if features.get("cdr_score") is not None:
                parts.append(f"  - CDR Score: {features['cdr_score']}")
            if features.get("nwbv"):
                parts.append(f"  - Brain Volume (nWBV): {features['nwbv']}")
        
        if "shap_values" in context and context["shap_values"]:
            parts.append("\nMost Important Factors:")
            sorted_shap = sorted(context["shap_values"].items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_shap:
                parts.append(f"  - {feature.replace('_', ' ').title()}: {importance*100:.1f}% importance")
        
        return "\n".join(parts)
    
    async def generate_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "technical"
    ) -> str:
        """
        Generate an LLM response.
        
        Args:
            user_message: The user's question or request
            context: Optional prediction context
            mode: "technical" or "patient"
        
        Returns:
            Generated response text
        """
        system_prompt = self._build_system_prompt(mode)
        context_info = self._build_context_prompt(context)
        
        full_prompt = f"""
{system_prompt}

=== CURRENT PREDICTION CONTEXT ===
{context_info}
=================================

User Question: {user_message}

Please provide a helpful response based on the context above."""
        
        # Try using Ollama
        client = self._get_client()
        
        if client:
            try:
                response = client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context_info}\n\nQuestion: {user_message}"}
                    ]
                )
                return response['message']['content']
            except Exception as e:
                print(f"⚠️ Ollama error: {e}. Using fallback response.")
        
        # Fallback response when Ollama is not available
        return self._generate_fallback_response(user_message, context, mode)
    
    def _generate_fallback_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]],
        mode: str
    ) -> str:
        """Generate a fallback response when LLM is unavailable."""
        if not context:
            return "I don't have specific prediction results to explain. Please run a prediction first."
        
        predicted_class = context.get("predicted_class", "Unknown")
        confidence = context.get("confidence_score", 0) * 100
        
        if mode == "technical":
            responses = {
                "NonDemented": f"""Based on the MRI analysis and clinical features, the AI model indicates 
a classification of **Cognitively Normal** with {confidence:.1f}% confidence.

The brain imaging shows typical age-appropriate structural patterns without significant 
hippocampal atrophy or ventricular enlargement. The cognitive markers, including MMSE scores, 
fall within normal ranges.

**Clinical Considerations:**
- Continue routine cognitive monitoring as per standard protocols
- Recommend follow-up assessment in 12-18 months
- Address modifiable risk factors (cardiovascular health, cognitive engagement)

This result should be interpreted alongside clinical examination and patient history.""",

                "VeryMildDemented": f"""The AI analysis suggests **Very Mild Cognitive Impairment** with 
{confidence:.1f}% confidence.

Imaging analysis reveals subtle changes that may indicate early neurodegenerative processes:
- Mild hippocampal volume reduction
- Early entorhinal cortex changes

**Clinical Recommendations:**
- Consider additional neuropsychological testing
- Evaluate for reversible causes of cognitive decline
- Discuss lifestyle modifications and cognitive rehabilitation
- Schedule follow-up in 6 months for monitoring progression

Further diagnostic workup may be warranted based on clinical judgment.""",

                "MildDemented": f"""The AI model classification indicates **Mild Dementia** consistent with 
early-stage Alzheimer's disease, with {confidence:.1f}% confidence.

Structural imaging features show:
- Moderate hippocampal atrophy
- Temporal lobe volume reduction
- Enlarged ventricles

**Clinical Pathway:**
- Comprehensive cognitive assessment recommended
- Consider CSF biomarkers or PET imaging for confirmation
- Discuss treatment options (cholinesterase inhibitors)
- Initiate caregiving support and future planning discussions
- Refer to memory clinic for specialized care

This is a screening result and requires clinical correlation.""",

                "ModerateDemented": f"""The analysis indicates **Moderate Dementia** with {confidence:.1f}% 
confidence, suggesting significant neurodegenerative changes.

Imaging features demonstrate:
- Marked hippocampal and temporal lobe atrophy
- Significant ventricular enlargement
- Cortical thinning patterns consistent with AD progression

**Immediate Considerations:**
- Urgent specialist referral if not already established
- Comprehensive care planning and safety assessment
- Caregiver support and resource coordination
- Medication optimization and symptom management

This patient likely requires structured support and monitoring."""
            }
        else:  # patient mode
            responses = {
                "NonDemented": f"""Great news! Based on the brain scan analysis, your results look **healthy 
and normal** (confidence: {confidence:.0f}%).

This means the computer didn't find any concerning patterns in your brain images. Your brain 
structure appears typical for your age.

**What this means:**
- This is a good result, but it's just one piece of the puzzle
- Continue doing the things that keep your brain healthy: exercise, staying socially 
  active, getting good sleep, and keeping your mind engaged
- Follow up with your doctor for routine check-ups

Remember, this is a screening tool - your doctor will give you the complete picture.""",

                "VeryMildDemented": f"""The analysis found some **very mild changes** in your brain scan 
(confidence: {confidence:.0f}%).

Please don't be alarmed - this is a very early finding and could have many explanations. 
Think of it like a gentle yellow light, suggesting it's worth paying a bit more attention.

**What you can do:**
- Talk to your doctor about these results
- There may be some additional tests recommended
- Focus on brain-healthy activities: physical exercise, mental stimulation, social connections
- Many people with similar results maintain their quality of life for many years

You're not facing this alone - there's lots of support available.""",

                "MildDemented": f"""The brain scan analysis suggests some **mild changes** that your 
doctor will want to discuss with you (confidence: {confidence:.0f}%).

I know this might feel concerning, but having this information early is actually helpful. 
It means you and your healthcare team can work together on the best path forward.

**Important next steps:**
- Schedule a detailed conversation with your doctor
- They may recommend some additional assessments
- There are treatments and strategies that can help
- Support groups and resources are available for you and your family

Remember: This is a screening tool, not a final diagnosis. Your doctor will explain 
everything in detail and answer all your questions.""",

                "ModerateDemented": f"""The analysis indicates some **significant changes** in the brain 
scan (confidence: {confidence:.0f}%).

I understand this may be difficult news. It's important to discuss these results with your 
healthcare team as soon as possible so they can provide proper care and support.

**What happens next:**
- Your doctor will explain these results and what they mean for you specifically
- There are many resources available to help you and your family
- Treatment options exist to help manage symptoms
- You don't have to navigate this alone

Please reach out to your healthcare provider to discuss these results and create a 
care plan together. Support is available every step of the way. 💙"""
            }
        
        return responses.get(predicted_class, "Results are being processed. Please consult with your healthcare provider.")
    
    async def get_explanation(
        self,
        prediction_result: Dict[str, Any],
        mode: str = "technical"
    ) -> str:
        """
        Generate a comprehensive explanation of prediction results.
        
        Args:
            prediction_result: Full prediction result dictionary
            mode: Explanation mode
        
        Returns:
            Detailed explanation string
        """
        prompt = "Please provide a comprehensive explanation of these prediction results, " \
                 "including what they mean and any relevant recommendations."
        
        return await self.generate_response(prompt, prediction_result, mode)
    
    def check_availability(self) -> Dict[str, Any]:
        """Check if the LLM service is available."""
        client = self._get_client()
        
        if client:
            try:
                models = client.list()
                return {
                    "available": True,
                    "host": self.ollama_host,
                    "model": self.model_name,
                    "installed_models": [m['name'] for m in models.get('models', [])]
                }
            except Exception as e:
                return {
                    "available": False,
                    "error": str(e),
                    "fallback": "Using built-in response templates"
                }
        
        return {
            "available": False,
            "error": "Ollama package not installed",
            "fallback": "Using built-in response templates"
        }


# Singleton
_llm_service = None

def get_llm_service() -> LLMService:
    """Get or create LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
