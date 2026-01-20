"""
CARE-AD+ RAG (Retrieval-Augmented Generation) Service

Provides medical knowledge base for enhanced LLM responses.
Implements vector similarity search for relevant context retrieval.
"""
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np


class MedicalKnowledgeBase:
    """
    Medical knowledge base for Alzheimer's disease.
    Contains clinical guidelines, staging criteria, and treatment information.
    """
    
    def __init__(self):
        self.knowledge = {
            "alzheimers_overview": {
                "definition": "Alzheimer's disease is a progressive neurodegenerative disorder characterized by cognitive decline, memory loss, and behavioral changes.",
                "prevalence": "Most common cause of dementia, affecting ~6.7 million Americans aged 65+",
                "pathology": "Characterized by amyloid-beta plaques and neurofibrillary tangles",
                "risk_factors": ["Age", "Family history", "APOE-e4 gene", "Cardiovascular disease", "Head trauma"]
            },
            
            "staging_cdr": {
                "CDR_0": {
                    "name": "Normal",
                    "description": "No cognitive impairment",
                    "memory": "No memory loss or slight inconsistent forgetfulness",
                    "orientation": "Fully oriented",
                    "clinical_class": "NonDemented"
                },
                "CDR_0.5": {
                    "name": "Very Mild Dementia",
                    "description": "Questionable dementia",
                    "memory": "Consistent slight forgetfulness, partial recollection of events",
                    "orientation": "Fully oriented except for slight difficulty with time relationships",
                    "clinical_class": "VeryMildDemented"
                },
                "CDR_1": {
                    "name": "Mild Dementia",
                    "description": "Mild cognitive decline",
                    "memory": "Moderate memory loss, more marked for recent events",
                    "orientation": "Moderate difficulty with time relationships, oriented for place",
                    "judgment": "Moderate difficulty handling problems",
                    "clinical_class": "MildDemented"
                },
                "CDR_2": {
                    "name": "Moderate Dementia",
                    "description": "Moderate cognitive decline",
                    "memory": "Severe memory loss, only highly learned material retained",
                    "orientation": "Severe difficulty with time relationships, usually disoriented to time",
                    "judgment": "Severely impaired",
                    "clinical_class": "ModerateDemented"
                }
            },
            
            "biomarkers": {
                "hippocampal_atrophy": {
                    "significance": "Early marker of Alzheimer's disease",
                    "description": "Reduction in hippocampal volume, particularly in medial temporal lobe",
                    "detection": "Visible on MRI scans",
                    "correlation": "Strongly correlated with memory impairment"
                },
                "ventricular_enlargement": {
                    "significance": "Indicates brain atrophy",
                    "description": "Expansion of cerebral ventricles due to brain tissue loss",
                    "detection": "Visible on structural MRI",
                    "progression": "Increases with disease severity"
                },
                "cortical_atrophy": {
                    "significance": "Widespread neurodegeneration",
                    "description": "Thinning of cerebral cortex, particularly temporal and parietal lobes",
                    "detection": "MRI volumetric analysis",
                    "stages": "Mild in early stages, severe in advanced disease"
                }
            },
            
            "clinical_recommendations": {
                "NonDemented": {
                    "follow_up": "Routine monitoring per standard care protocols",
                    "lifestyle": ["Regular physical exercise", "Cognitive stimulation", "Heart-healthy diet", "Social engagement"],
                    "monitoring": "Annual cognitive screening for at-risk individuals",
                    "prevention": "Maintain cardiovascular health, control blood pressure and diabetes"
                },
                "VeryMildDemented": {
                    "follow_up": "Comprehensive neuropsychological evaluation within 3 months",
                    "referral": "Consider memory clinic referral",
                    "testing": ["CSF biomarkers", "Amyloid PET scan", "Detailed cognitive battery"],
                    "monitoring": "6-month follow-up imaging",
                    "counseling": "Patient and family education about findings"
                },
                "MildDemented": {
                    "follow_up": "Urgent neurology/geriatric referral",
                    "pharmacological": ["Cholinesterase inhibitors (donepezil, rivastigmine)", "Memantine consideration"],
                    "non_pharmacological": ["Cognitive rehabilitation", "Occupational therapy", "Support groups"],
                    "safety": "Assess driving ability, home safety, financial management",
                    "care_planning": "Initiate advance care planning discussions"
                },
                "ModerateDemented": {
                    "follow_up": "Immediate specialist consultation",
                    "medication_review": "Optimize current treatments, manage behavioral symptoms",
                    "safety_assessment": "Comprehensive safety evaluation required",
                    "caregiver_support": ["Respite care", "Support groups", "Education programs"],
                    "living_arrangements": "Evaluate need for supervised living",
                    "legal": "Power of attorney, healthcare proxy if not already established"
                }
            },
            
            "imaging_findings": {
                "mri_patterns": {
                    "normal": "Preserved brain volume, no significant atrophy, normal ventricle size",
                    "early_ad": "Subtle hippocampal volume loss, mild temporal lobe changes",
                    "mild_ad": "Moderate hippocampal atrophy, temporal lobe atrophy, mild ventricular enlargement",
                    "moderate_ad": "Marked hippocampal atrophy, generalized cortical atrophy, significant ventricular dilation"
                },
                "differential_diagnosis": {
                    "vascular_dementia": "Multiple infarcts, white matter changes, preserved hippocampus",
                    "frontotemporal_dementia": "Frontal and temporal atrophy, relative hippocampal sparing",
                    "lewy_body_dementia": "Preserved medial temporal structures, occipital hypoperfusion"
                }
            },
            
            "treatment_options": {
                "pharmacological": {
                    "cholinesterase_inhibitors": {
                        "drugs": ["Donepezil (Aricept)", "Rivastigmine (Exelon)", "Galantamine (Razadyne)"],
                        "mechanism": "Increase acetylcholine levels in brain",
                        "efficacy": "Modest improvement in cognition and function",
                        "stages": "Mild to moderate Alzheimer's"
                    },
                    "nmda_antagonist": {
                        "drug": "Memantine (Namenda)",
                        "mechanism": "Regulates glutamate activity",
                        "efficacy": "May slow progression in moderate to severe AD",
                        "combination": "Can be combined with cholinesterase inhibitors"
                    }
                },
                "non_pharmacological": {
                    "cognitive_training": "Structured programs to maintain cognitive function",
                    "physical_exercise": "30 minutes daily, shown to slow cognitive decline",
                    "social_engagement": "Regular social activities and interactions",
                    "nutrition": "Mediterranean diet, omega-3 fatty acids"
                }
            }
        }
    
    def get_relevant_context(
        self,
        predicted_class: str,
        query_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Retrieve relevant medical knowledge for a prediction.
        
        Args:
            predicted_class: The predicted disease stage
            query_type: Type of information needed (general, treatment, imaging, etc.)
        
        Returns:
            Dictionary of relevant medical knowledge
        """
        context = {}
        
        # Add staging information
        cdr_mapping = {
            "NonDemented": "CDR_0",
            "VeryMildDemented": "CDR_0.5",
            "MildDemented": "CDR_1",
            "ModerateDemented": "CDR_2"
        }
        
        cdr_stage = cdr_mapping.get(predicted_class)
        if cdr_stage:
            context["staging"] = self.knowledge["staging_cdr"][cdr_stage]
        
        # Add clinical recommendations
        if predicted_class in self.knowledge["clinical_recommendations"]:
            context["recommendations"] = self.knowledge["clinical_recommendations"][predicted_class]
        
        # Add biomarker information
        context["biomarkers"] = self.knowledge["biomarkers"]
        
        # Add treatment options if applicable
        if predicted_class in ["MildDemented", "ModerateDemented"]:
            context["treatment"] = self.knowledge["treatment_options"]
        
        # Add imaging patterns
        context["imaging"] = self.knowledge["imaging_findings"]["mri_patterns"]
        
        return context
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search through knowledge base.
        In production, use vector embeddings and semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant knowledge entries
        """
        query_lower = query.lower()
        results = []
        
        # Search through all knowledge entries
        for category, content in self.knowledge.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, dict):
                        # Check if query terms appear in the content
                        content_str = json.dumps(value).lower()
                        if any(term in content_str for term in query_lower.split()):
                            results.append({
                                "category": category,
                                "key": key,
                                "content": value,
                                "relevance": self._calculate_relevance(query_lower, content_str)
                            })
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Simple relevance scoring based on term frequency."""
        query_terms = query.split()
        score = sum(content.count(term) for term in query_terms)
        return score


class RAGService:
    """
    RAG service that combines knowledge retrieval with LLM generation.
    """
    
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
    
    def enhance_prompt(
        self,
        message: str,
        predicted_class: str,
        mode: str = "technical"
    ) -> str:
        """
        Enhance LLM prompt with relevant medical knowledge.
        
        Args:
            message: User's question
            predicted_class: Predicted disease stage
            mode: Response mode (technical or patient)
        
        Returns:
            Enhanced prompt with knowledge context
        """
        # Get relevant context
        context = self.knowledge_base.get_relevant_context(predicted_class)
        
        # Search for query-specific knowledge
        search_results = self.knowledge_base.search_knowledge(message)
        
        # Build enhanced prompt
        prompt_parts = []
        
        if mode == "technical":
            prompt_parts.append("You are a clinical decision support AI with access to medical knowledge.")
        else:
            prompt_parts.append("You are a caring medical assistant with medical knowledge.")
        
        # Add staging information
        if "staging" in context:
            staging = context["staging"]
            prompt_parts.append(f"\nClinical Stage: {staging['name']} (CDR {staging.get('cdr', 'N/A')})")
            prompt_parts.append(f"Description: {staging['description']}")
        
        # Add recommendations
        if "recommendations" in context:
            recs = context["recommendations"]
            prompt_parts.append("\nClinical Recommendations:")
            for key, value in recs.items():
                if isinstance(value, list):
                    prompt_parts.append(f"- {key}: {', '.join(value)}")
                else:
                    prompt_parts.append(f"- {key}: {value}")
        
        # Add search results
        if search_results:
            prompt_parts.append("\nRelevant Medical Knowledge:")
            for result in search_results:
                prompt_parts.append(f"- {result['category']}/{result['key']}: {json.dumps(result['content'], indent=2)}")
        
        prompt_parts.append(f"\nUser Question: {message}")
        prompt_parts.append("\nProvide a comprehensive, evidence-based response:")
        
        return "\n".join(prompt_parts)
    
    def get_context_for_prediction(self, predicted_class: str) -> Dict[str, Any]:
        """Get full context for a prediction."""
        return self.knowledge_base.get_relevant_context(predicted_class)


# Singleton
_rag_service = None

def get_rag_service() -> RAGService:
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
