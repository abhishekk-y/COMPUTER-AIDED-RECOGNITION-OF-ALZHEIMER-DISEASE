"""
CARE-AD+ Clinical Report Service

Generates professional PDF reports for clinical documentation.
Reports include prediction results, XAI visualizations, and LLM explanations.
"""
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from app.config import settings


class ReportService:
    """
    Service for generating clinical PDF reports.
    """
    
    def __init__(self):
        self.reports_dir = settings.REPORTS_DIR
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Custom styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1F2937')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#4F46E5'),
            borderColor=colors.HexColor('#4F46E5'),
            borderWidth=0,
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='ResultHighlight',
            parent=self.styles['Normal'],
            fontSize=16,
            alignment=TA_CENTER,
            spaceBefore=10,
            spaceAfter=10,
            textColor=colors.HexColor('#1F2937'),
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#6B7280')
        ))
    
    def _get_result_color(self, predicted_class: str) -> colors.Color:
        """Get color based on prediction result."""
        color_map = {
            "NonDemented": colors.HexColor('#10B981'),  # Green
            "VeryMildDemented": colors.HexColor('#F59E0B'),  # Amber
            "MildDemented": colors.HexColor('#F97316'),  # Orange
            "ModerateDemented": colors.HexColor('#EF4444')  # Red
        }
        return color_map.get(predicted_class, colors.HexColor('#6B7280'))
    
    def generate_clinical_report(
        self,
        prediction,  # Prediction model instance
        patient,     # Patient model instance
        include_gradcam: bool = True,
        include_shap: bool = True,
        include_llm_explanation: bool = True
    ) -> str:
        """
        Generate a comprehensive clinical PDF report.
        
        Args:
            prediction: Prediction ORM object
            patient: Patient ORM object
            include_gradcam: Whether to include Grad-CAM visualization
            include_shap: Whether to include SHAP analysis
            include_llm_explanation: Whether to include LLM explanation
        
        Returns:
            Path to the generated PDF file
        """
        # Generate unique filename
        report_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CARE_AD_Report_{patient.patient_id}_{timestamp}_{report_id}.pdf"
        filepath = os.path.join(self.reports_dir, filename)
        
        # Create document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Build story (content)
        story = []
        
        # Header with logo placeholder
        story.append(self._create_header(report_id))
        story.append(Spacer(1, 20))
        
        # Patient information
        story.append(self._create_patient_section(patient))
        story.append(Spacer(1, 15))
        
        # Prediction results
        story.append(self._create_prediction_section(prediction))
        story.append(Spacer(1, 15))
        
        # Probability breakdown
        if prediction.probabilities:
            story.append(self._create_probability_section(prediction))
            story.append(Spacer(1, 15))
        
        # Grad-CAM visualization
        if include_gradcam and prediction.gradcam_path and os.path.exists(prediction.gradcam_path):
            story.append(self._create_gradcam_section(prediction.gradcam_path))
            story.append(Spacer(1, 15))
        
        # SHAP analysis
        if include_shap and prediction.clinical_features:
            story.append(self._create_shap_section(prediction))
            story.append(Spacer(1, 15))
        
        # LLM Explanation
        if include_llm_explanation:
            story.append(self._create_explanation_section(prediction))
            story.append(Spacer(1, 15))
        
        # Clinical recommendations
        story.append(self._create_recommendations_section(prediction.predicted_class))
        story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(self._create_disclaimer_section())
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(self._create_footer(report_id))
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ Report generated: {filepath}")
        return filepath
    
    def _create_header(self, report_id: str):
        """Create report header."""
        header_data = [
            [
                Paragraph(f"<b>{settings.INSTITUTION_NAME}</b>", self.styles['ReportTitle']),
            ],
            [
                Paragraph("CARE-AD+ Clinical Assessment Report", 
                         ParagraphStyle('Subtitle', fontSize=12, alignment=TA_CENTER, 
                                       textColor=colors.HexColor('#6B7280')))
            ],
            [
                Paragraph(f"Report ID: {report_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                         ParagraphStyle('ReportMeta', fontSize=9, alignment=TA_CENTER,
                                       textColor=colors.HexColor('#9CA3AF')))
            ]
        ]
        
        table = Table(header_data, colWidths=[doc_width := 17*cm])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        
        return table
    
    def _create_patient_section(self, patient):
        """Create patient demographics section."""
        elements = [
            Paragraph("Patient Information", self.styles['SectionHeading']),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB')),
            Spacer(1, 10)
        ]
        
        # Patient info table
        data = [
            ["Patient ID:", patient.patient_id, "Age:", f"{patient.age or 'N/A'} years"],
            ["Gender:", patient.gender or "N/A", "Education:", f"{patient.education_years or 'N/A'} years"],
            ["MMSE Score:", f"{patient.mmse_score or 'N/A'}/30", "CDR Score:", f"{patient.cdr_score or 'N/A'}"],
        ]
        
        table = Table(data, colWidths=[3*cm, 4*cm, 3*cm, 4*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4B5563')),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#4B5563')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        return elements
    
    def _create_prediction_section(self, prediction):
        """Create prediction results section."""
        result_color = self._get_result_color(prediction.predicted_class)
        
        elements = [
            Paragraph("AI Prediction Result", self.styles['SectionHeading']),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB')),
            Spacer(1, 15)
        ]
        
        # Result box
        result_text = self._get_class_description(prediction.predicted_class)
        confidence = prediction.confidence_score * 100
        
        result_data = [
            [Paragraph(f"<b>{prediction.predicted_class}</b>", 
                      ParagraphStyle('Result', fontSize=18, alignment=TA_CENTER,
                                    textColor=result_color))],
            [Paragraph(result_text, 
                      ParagraphStyle('ResultDesc', fontSize=10, alignment=TA_CENTER,
                                    textColor=colors.HexColor('#4B5563')))],
            [Paragraph(f"Confidence: <b>{confidence:.1f}%</b>",
                      ParagraphStyle('Confidence', fontSize=12, alignment=TA_CENTER,
                                    textColor=colors.HexColor('#1F2937')))]
        ]
        
        result_table = Table(result_data, colWidths=[12*cm])
        result_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOX', (0, 0), (-1, -1), 2, result_color),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F9FAFB')),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        elements.append(result_table)
        return elements
    
    def _get_class_description(self, predicted_class: str) -> str:
        """Get human-readable description for class."""
        descriptions = {
            "NonDemented": "Cognitively Normal - No significant signs of dementia detected",
            "VeryMildDemented": "Very Mild Cognitive Impairment - Early changes detected",
            "MildDemented": "Mild Dementia - Consistent with early-stage Alzheimer's disease",
            "ModerateDemented": "Moderate Dementia - Significant cognitive impairment detected"
        }
        return descriptions.get(predicted_class, predicted_class)
    
    def _create_probability_section(self, prediction):
        """Create probability breakdown section."""
        elements = [
            Paragraph("Classification Probabilities", self.styles['SectionHeading']),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB')),
            Spacer(1, 10)
        ]
        
        probs = json.loads(prediction.probabilities) if isinstance(prediction.probabilities, str) else prediction.probabilities
        
        data = [["Class", "Probability", "Visual"]]
        for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar_width = int(prob * 100)
            bar = "█" * (bar_width // 5) + "░" * ((100 - bar_width) // 5)
            data.append([class_name, f"{prob*100:.1f}%", bar])
        
        table = Table(data, colWidths=[5*cm, 3*cm, 8*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3F4F6')),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        return elements
    
    def _create_gradcam_section(self, gradcam_path: str):
        """Create Grad-CAM visualization section."""
        elements = [
            Paragraph("Brain Region Analysis (Grad-CAM)", self.styles['SectionHeading']),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB')),
            Spacer(1, 10),
            Paragraph("The highlighted regions show areas of the brain that most influenced the AI's prediction:",
                     self.styles['BodyText']),
            Spacer(1, 10)
        ]
        
        try:
            img = Image(gradcam_path, width=15*cm, height=5*cm)
            img.hAlign = 'CENTER'
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"[Visualization unavailable: {e}]", self.styles['BodyText']))
        
        return elements
    
    def _create_shap_section(self, prediction):
        """Create SHAP feature importance section."""
        elements = [
            Paragraph("Clinical Feature Importance", self.styles['SectionHeading']),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB')),
            Spacer(1, 10),
            Paragraph("The following clinical features contributed to the prediction:",
                     self.styles['BodyText']),
            Spacer(1, 5)
        ]
        
        if prediction.shap_values:
            shap = json.loads(prediction.shap_values) if isinstance(prediction.shap_values, str) else prediction.shap_values
            clinical = json.loads(prediction.clinical_features) if prediction.clinical_features else {}
            
            data = [["Feature", "Value", "Importance"]]
            for feature, importance in sorted(shap.items(), key=lambda x: x[1], reverse=True):
                value = clinical.get(feature, "N/A")
                data.append([feature.replace('_', ' ').title(), str(value), f"{importance*100:.1f}%"])
            
            table = Table(data, colWidths=[6*cm, 4*cm, 4*cm])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3F4F6')),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(table)
        
        return elements
    
    def _create_explanation_section(self, prediction):
        """Create LLM explanation section."""
        elements = [
            Paragraph("AI-Generated Clinical Summary", self.styles['SectionHeading']),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB')),
            Spacer(1, 10)
        ]
        
        explanation = prediction.llm_explanation_technical or self._get_default_explanation(prediction.predicted_class)
        
        # Split into paragraphs
        for para in explanation.split('\n\n'):
            if para.strip():
                elements.append(Paragraph(para.strip(), self.styles['BodyText']))
                elements.append(Spacer(1, 5))
        
        return elements
    
    def _get_default_explanation(self, predicted_class: str) -> str:
        """Get default explanation when LLM not available."""
        explanations = {
            "NonDemented": """The AI analysis indicates normal cognitive function with no significant 
signs of dementia. Brain structure appears typical for the patient's age group, with preserved 
hippocampal volume and normal ventricular size. Cognitive assessment scores fall within expected 
ranges. Continue routine monitoring and maintain brain-healthy lifestyle factors.""",

            "VeryMildDemented": """The AI analysis suggests very mild cognitive changes. There may be 
subtle structural changes in brain regions associated with memory, such as the hippocampus and 
entorhinal cortex. These findings warrant close monitoring and may benefit from additional 
neuropsychological evaluation. Consider lifestyle modifications and cognitive engagement activities.""",

            "MildDemented": """The AI analysis indicates mild dementia consistent with early-stage 
Alzheimer's disease. Structural imaging shows moderate hippocampal atrophy and temporal lobe 
changes. Clinical cognitive assessments reveal measurable decline. A comprehensive evaluation 
and care plan development is recommended, including discussion of treatment options.""",

            "ModerateDemented": """The AI analysis indicates moderate dementia with significant 
structural brain changes including marked hippocampal atrophy, ventricular enlargement, and 
cortical thinning. Specialist consultation is strongly recommended for care planning, treatment 
optimization, and caregiver support coordination."""
        }
        return explanations.get(predicted_class, "Detailed explanation not available.")
    
    def _create_recommendations_section(self, predicted_class: str):
        """Create clinical recommendations section."""
        elements = [
            Paragraph("Recommendations", self.styles['SectionHeading']),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB')),
            Spacer(1, 10)
        ]
        
        recommendations = {
            "NonDemented": [
                "• Continue routine cognitive health monitoring",
                "• Maintain cardiovascular health and physical activity",
                "• Engage in cognitively stimulating activities",
                "• Follow up as per standard care protocols"
            ],
            "VeryMildDemented": [
                "• Schedule comprehensive neuropsychological evaluation",
                "• Consider referral to memory clinic",
                "• Evaluate for reversible causes of cognitive decline",
                "• Discuss cognitive rehabilitation options",
                "• Follow up in 6 months for repeat assessment"
            ],
            "MildDemented": [
                "• Urgent referral to neurologist/geriatrician",
                "• Consider pharmacological treatment options",
                "• Initiate care planning discussions",
                "• Assess functional abilities and safety",
                "• Connect with caregiver support resources",
                "• Consider additional biomarker testing"
            ],
            "ModerateDemented": [
                "• Immediate specialist consultation required",
                "• Comprehensive safety assessment",
                "• Structured care plan development",
                "• Caregiver education and support",
                "• Evaluate for clinical trials if appropriate",
                "• Coordinate with social services as needed"
            ]
        }
        
        for rec in recommendations.get(predicted_class, ["Consult with healthcare provider"]):
            elements.append(Paragraph(rec, self.styles['BodyText']))
        
        return elements
    
    def _create_disclaimer_section(self):
        """Create disclaimer section."""
        disclaimer_text = """
        <b>IMPORTANT DISCLAIMER:</b> This report has been generated by CARE-AD+, an AI-powered 
        clinical decision support system. The predictions and visualizations contained herein 
        are intended to assist healthcare professionals in their clinical assessment and should 
        NOT be used as a sole basis for diagnosis or treatment decisions. All findings must be 
        interpreted in conjunction with clinical examination, patient history, and other diagnostic 
        tests. This system is designed as a screening tool and does not replace professional 
        medical judgment. Healthcare providers should verify all information before making 
        clinical decisions. Patient data has been processed in accordance with applicable 
        privacy regulations.
        """
        
        return [
            HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB')),
            Spacer(1, 10),
            Paragraph(disclaimer_text, self.styles['Disclaimer'])
        ]
    
    def _create_footer(self, report_id: str):
        """Create report footer."""
        footer_text = f"""
        {settings.INSTITUTION_NAME} | CARE-AD+ Clinical Assessment System
        Report ID: {report_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Confidential Medical Document - Handle According to Institutional Protocols
        """
        
        return Paragraph(
            footer_text,
            ParagraphStyle('Footer', fontSize=8, alignment=TA_CENTER,
                          textColor=colors.HexColor('#9CA3AF'))
        )


# Singleton
_report_service = None

def get_report_service() -> ReportService:
    """Get or create report service singleton."""
    global _report_service
    if _report_service is None:
        _report_service = ReportService()
    return _report_service
