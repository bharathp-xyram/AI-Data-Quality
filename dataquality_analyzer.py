#!/usr/bin/env python3
"""
Part 1: Imports and Pure LLM Clinical Analyzer
==============================================

This part contains all imports and the GenerativeClinicalAnalyzer class
that handles pure LLM communication without any knowledge base.
"""

# Standard library imports
import json
import os
import re
import sys
from datetime import datetime, time
from typing import Dict, List, Any, Optional

import anthropic
# Third-party imports
import openai
from lxml import etree

# Check for optional dependencies
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Optional dependencies not available. Pure LLM mode only.")


class GenerativeClinicalAnalyzer:
    """
    Pure LLM Clinical Analyzer - NO knowledge base, NO reference ranges.

    Uses direct LLM communication for ALL clinical assessments including:
    - Drug interaction detection
    - Clinical plausibility assessment
    - Completeness evaluation
    - Structural integrity analysis
    - Narrative consistency checking
    """

    def __init__(self, provider="openai", api_key=None, model_name=None):
        """Initialize pure LLM analyzer."""
        self.provider = provider
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')

        if provider == "openai":
            self.model_name = model_name or "gpt-4o-mini"
            if self.api_key:
                openai.api_key = self.api_key
            self.claude_client = None  # Not using Claude

        elif provider == "claude":
            self.model_name = model_name or "claude-3-5-sonnet-20241022"
            # ✅ ADD CLAUDE CLIENT INITIALIZATION:
            if self.api_key:
                try:
                    import anthropic
                    self.claude_client = anthropic.Anthropic(api_key=self.api_key)
                    print(f"✅ Claude client initialized successfully")
                except ImportError:
                    print("❌ Anthropic library not available")
                    self.claude_client = None
                except Exception as e:
                    print(f"❌ Claude client initialization failed: {e}")
                    self.claude_client = None
            else:
                self.claude_client = None

        self.enabled = self.api_key is not None and (provider != "claude" or self.claude_client is not None)
        print(f" LLM Clinical Analyzer: {provider} - {'✅ Enabled' if self.enabled else '❌ Disabled'}")

    def analyze_section_comprehensive(self, structured_data: List[Dict], narrative_text: str,
                                      section_type: str, analysis_dimension: str,
                                      patient_age: int = None, patient_gender: str = None) -> Dict[str, Any]:
        """
        Pure LLM analysis for any section and dimension.

        NO reference ranges, NO knowledge base - pure AI reasoning.

        Args:
            structured_data: Clinical data items
            narrative_text: Narrative text
            section_type: 'lab_results', 'medications', 'problems'
            analysis_dimension: 'clinical_plausibility', 'completeness', 'structural_integrity', 'narrative_consistency'
            patient_age: Patient age for context
            patient_gender: Patient gender for context

        Returns:
            Dict with LLM analysis results
        """
        if not self.enabled:
            return self._fallback_response(section_type, analysis_dimension)

        # Build pure LLM question
        question = self._build_pure_llm_question(
            structured_data, narrative_text, section_type, analysis_dimension, patient_age, patient_gender
        )

        try:
            if self.provider == "openai":
                return self._query_openai(question, section_type, analysis_dimension)
            elif self.provider == "claude":
                return self._query_claude(question, section_type, analysis_dimension)
        except Exception as e:
            print(f"⚠️ LLM query error: {e}")
            return self._fallback_response(section_type, analysis_dimension)

    def analyze_lab_direct(self, lab_name: str, value: str, unit: str,
                           patient_age: int = None, patient_gender: str = None) -> Dict[str, Any]:
        """
        Direct LLM lab analysis - NO reference ranges.

        Args:
            lab_name: Laboratory test name
            value: Test result value
            unit: Unit of measurement
            patient_age: Patient age for context
            patient_gender: Patient gender for context

        Returns:
            Dict with clinical assessment
        """
        if not self.enabled:
            return self._fallback_lab_response(lab_name, value, unit)

        # Build pure clinical question
        question = self._build_pure_lab_question(lab_name, value, unit, patient_age, patient_gender)

        try:
            if self.provider == "openai":
                return self._query_openai_lab(question, lab_name, value, unit)
            elif self.provider == "claude":
                return self._query_claude_lab(question, lab_name, value, unit)
        except Exception as e:
            print(f"⚠️ LLM lab query error: {e}")
            return self._fallback_lab_response(lab_name, value, unit)

    def _build_pure_llm_question(self, structured_data: List[Dict], narrative_text: str,
                                 section_type: str, analysis_dimension: str,
                                 patient_age: int = None, patient_gender: str = None) -> str:
        """Build pure LLM question without any knowledge base."""

        # Patient context
        patient_context = "Patient"
        if patient_age and patient_gender:
            patient_context = f"{patient_age}-year-old {patient_gender.lower()} patient"
        elif patient_age:
            patient_context = f"{patient_age}-year-old patient"

        # Section context
        section_name = section_type.replace('_', ' ').title()

        # Build data summary
        data_summary = self._build_data_summary(structured_data, section_type)

        # Narrative context
        narrative_summary = f"Narrative text: {narrative_text[:300]}..." if narrative_text else "No narrative text available"

        #  LLM questions (no reference ranges mentioned)
        dimension_questions = self._get_pure_llm_questions(analysis_dimension, section_type)

        question = f"""Clinical Assessment Request - {section_name} Section:

{patient_context} clinical documentation analysis:

SECTION DATA:
{data_summary}

NARRATIVE:
{narrative_summary}

ANALYSIS DIMENSION: {analysis_dimension.replace('_', ' ').title()}

{dimension_questions}

Use your clinical knowledge to assess this data. DO NOT ask for reference ranges - use your medical training.

Respond in this exact JSON format:
{{
    "overall_assessment": "EXCELLENT|GOOD|FAIR|POOR|CRITICAL",
    "score": 0-100,
    "clinical_findings": ["finding1", "finding2", "finding3"],
    "recommendations": ["recommendation1", "recommendation2"],
    "specific_issues": ["issue1", "issue2"],
    "confidence": 0-100,
    "rationale": "brief explanation of assessment"
}}"""

        return question

    def _build_pure_lab_question(self, lab_name: str, value: str, unit: str,
                                 patient_age: int = None, patient_gender: str = None) -> str:
        """Build pure LLM lab question without reference ranges."""

        # Patient context
        patient_context = "Patient"
        if patient_age and patient_gender:
            patient_context = f"{patient_age}-year-old {patient_gender.lower()} patient"
        elif patient_age:
            patient_context = f"{patient_age}-year-old patient"

        question = f"""Clinical Lab Assessment:

{patient_context} has laboratory result:
- Test: {lab_name}
- Value: {value} {unit}

Using your clinical knowledge, assess this lab result:

1. Is this result NORMAL, ABNORMAL, or CRITICAL?
2. What is the clinical significance?
3. What immediate actions are needed?
4. What is the risk level?

Use your medical training to determine normal ranges. DO NOT ask for reference values.

Respond in this exact JSON format:
{{
    "status": "NORMAL|ABNORMAL|CRITICAL",
    "risk_level": "LOW|MODERATE|HIGH|CRITICAL",
    "clinical_significance": "brief explanation",
    "immediate_actions": ["action1", "action2"],
    "severity_score": 0-100,
    "confidence": 0-100
}}"""

        return question

    def _build_data_summary(self, structured_data: List[Dict], section_type: str) -> str:
        """Build data summary with code information for LLM analysis."""

        if not structured_data:
            return "No structured data available"

        summary_parts = []

        for i, item in enumerate(structured_data[:10]):  # Limit to first 10 items
            if section_type == 'lab_results':
                name = item.get('display_name', 'Unknown lab')
                value = item.get('value', 'No value')
                unit = item.get('unit', '')
                code = item.get('code')
                code_system = item.get('code_system')
                code_system_name = item.get('code_system_name')

                # Include code information for LLM analysis
                summary_parts.append(f"- {name}: {value} {unit}")
                if code:
                    code_info = f"Code: {code}"
                    if code_system_name:
                        code_info += f" ({code_system_name})"
                    if code_system:
                        code_info += f" [System: {code_system}]"
                    summary_parts.append(f"  {code_info}")

            elif section_type == 'medications':
                name = item.get('display_name', 'Unknown medication')
                dose = item.get('dose', 'No dose')
                route = item.get('route', 'No route')
                code = item.get('code')
                code_system = item.get('code_system')
                code_system_name = item.get('code_system_name')

                # Include code information for LLM analysis
                summary_parts.append(f"- {name}, {dose}, {route}")
                if code:
                    code_info = f"Code: {code}"
                    if code_system_name:
                        code_info += f" ({code_system_name})"
                    if code_system:
                        code_info += f" [System: {code_system}]"
                    summary_parts.append(f"  {code_info}")

            elif section_type == 'problems':
                name = item.get('display_name', 'Unknown problem')
                status = item.get('status', 'No status')
                code = item.get('code')
                code_system = item.get('code_system')
                code_system_name = item.get('code_system_name')

                # Include code information for LLM analysis
                summary_parts.append(f"- {name} ({status})")
                if code:
                    code_info = f"Code: {code}"
                    if code_system_name:
                        code_info += f" ({code_system_name})"
                    if code_system:
                        code_info += f" [System: {code_system}]"
                    summary_parts.append(f"  {code_info}")

        if len(structured_data) > 10:
            summary_parts.append(f"... and {len(structured_data) - 10} more items")

        return "\n".join(summary_parts) if summary_parts else "No valid structured data"

    def _get_pure_llm_questions(self, dimension: str, section_type: str) -> str:
        """Get pure LLM questions with code validation using only LLM medical knowledge."""

        questions = {
            'clinical_plausibility': {
                'lab_results': """
    CLINICAL PLAUSIBILITY ASSESSMENT:
    Use your clinical knowledge to assess these laboratory values:
    1. Are these laboratory values clinically reasonable for this patient?
    2. Do any values indicate critical conditions requiring immediate intervention?
    3. Are there any impossible or clearly erroneous values?
    4. Do the lab results make clinical sense together?
    5. Are these values appropriate for the patient's age?
    6. Do you recognize any patterns that suggest specific conditions?

    MEDICAL CODING ASSESSMENT:
    Use your medical coding knowledge to also assess:
    7. Do laboratory tests have appropriate LOINC codes where expected?
    8. Are code systems (like 2.16.840.1.113883.6.1 for LOINC) correctly used?
    9. Do display names match the clinical and coding context?
    10. Are there obvious coding errors or mismatches?
                """,
                'medications': """
    MEDICATION & DRUG INTERACTION ASSESSMENT:
    Use your pharmacological knowledge to assess this medication list:
    1. Are these medications clinically appropriate for this patient?
    2. Are there any dangerous drug interactions between these medications?
    3. Are dosages appropriate and realistic for the patient's age?
    4. Are there any age-inappropriate medications?
    5. Do the medications make sense together clinically?
    6. Are there duplicate therapies or therapeutic conflicts?
    7. Are there high-risk combinations requiring special monitoring?
    8. Do you recognize any contraindicated combinations?

    MEDICAL CODING ASSESSMENT:
    Use your medical coding knowledge to also assess:
    9. Do medications have appropriate RxNorm codes where expected?
    10. Are code systems (like 2.16.840.1.113883.6.88 for RxNorm) correctly used?
    11. Do display names match the medication and coding context?
    12. Are there obvious coding errors or inappropriate code usage?
                """,
                'problems': """
    DIAGNOSTIC PLAUSIBILITY ASSESSMENT:
    Use your clinical knowledge to assess this problem list:
    1. Are these diagnoses clinically consistent and plausible?
    2. Are there any conflicting diagnoses that cannot coexist?
    3. Are the diagnoses age-appropriate for this patient?
    4. Do the diagnoses make clinical sense together?
    5. Are there any vague or imprecise diagnoses?
    6. Do you recognize any diagnostic patterns or syndromes?

    MEDICAL CODING ASSESSMENT:
    Use your medical coding knowledge to also assess:
    7. Do diagnoses have appropriate SNOMED CT or ICD codes where expected?
    8. Are code systems (like 2.16.840.1.113883.6.96 for SNOMED CT) correctly used?
    9. Do display names match the diagnostic and coding context?
    10. Are there obvious coding errors or inappropriate diagnostic codes?
                """
            },
            'completeness': {
                'lab_results': """
    COMPLETENESS ASSESSMENT:
    Based on your clinical experience, assess completeness:
    1. Is this laboratory panel comprehensive for clinical decision-making?
    2. Are important labs likely missing for this patient's age and presentation?
    3. Do all lab entries have necessary components (names, values, units)?
    4. Is the quantity of tests appropriate for comprehensive assessment?
    5. What additional labs might be clinically indicated?

    CODING COMPLETENESS ASSESSMENT:
    Use your medical coding knowledge to assess:
    6. Do laboratory tests have proper medical codes (LOINC expected for labs)?
    7. Are code systems properly specified for interoperability?
    8. Do entries have clear, standardized names for clinical communication?
    9. What coding elements are missing that would improve data quality?
                """,
                'medications': """
    COMPLETENESS ASSESSMENT:
    Based on your clinical experience, assess completeness:
    1. Does this medication list appear comprehensive?
    2. Do all medications have necessary prescribing details (names, doses, routes)?
    3. Are there likely missing medications for common conditions?
    4. Is medication reconciliation adequate?
    5. What medications might be missing based on the diagnoses?

    CODING COMPLETENESS ASSESSMENT:
    Use your medical coding knowledge to assess:
    6. Do medications have proper medical codes (RxNorm expected for drugs)?
    7. Are code systems properly specified for medication identification?
    8. Do entries have standardized drug names for safety and interoperability?
    9. What coding elements are missing that would improve medication safety?
                """,
                'problems': """
    COMPLETENESS ASSESSMENT:
    Based on your clinical experience, assess completeness:
    1. Does this problem list appear comprehensive for this patient?
    2. Are all relevant diagnoses likely included?
    3. Do all problems have appropriate status information?
    4. Are chronic conditions properly documented?
    5. What diagnoses might be missing based on medications or labs?

    CODING COMPLETENESS ASSESSMENT:
    Use your medical coding knowledge to assess:
    6. Do diagnoses have proper medical codes (SNOMED CT/ICD expected)?
    7. Are code systems properly specified for diagnostic accuracy?
    8. Do entries have standardized diagnostic names for clinical communication?
    9. What coding elements are missing that would improve diagnostic clarity?
                """
            },
            'structural_integrity': {
                'lab_results': """
    STRUCTURAL INTEGRITY ASSESSMENT:
    Assess the organization, formatting, and data quality:
    1. Are lab entries properly formatted and structured?
    2. Do all entries have complete and accurate naming?
    3. Are values in appropriate formats with proper units?
    4. Is data organization logical and consistent?
    5. Are there obvious data entry errors or inconsistencies?

    CODING STRUCTURE ASSESSMENT:
    Use your medical coding knowledge to assess:
    6. Are medical codes properly formatted (LOINC codes should be numeric-dash-numeric)?
    7. Do code systems follow standard medical coding OID patterns?
    8. Are display names consistent with expected medical terminology?
    9. Do you see obvious code-to-name mismatches or formatting errors?
    10. Are codes appropriate for the type of clinical data (labs should use LOINC)?
                """,
                'medications': """
    STRUCTURAL INTEGRITY ASSESSMENT:
    Assess the organization, formatting, and data quality:
    1. Are medication entries properly formatted and structured?
    2. Do all medications have clear, unambiguous names?
    3. Are dosing instructions clear and complete?
    4. Is medication data organized consistently?
    5. Are there formatting issues or incomplete entries?

    CODING STRUCTURE ASSESSMENT:
    Use your medical coding knowledge to assess:
    6. Are medical codes properly formatted for medication identification?
    7. Do code systems follow standard medical coding patterns (RxNorm for drugs)?
    8. Are display names consistent with expected pharmaceutical terminology?
    9. Do you see obvious code-to-name mismatches or inappropriate codes?
    10. Are codes suitable for medication identification and safety?
                """,
                'problems': """
    STRUCTURAL INTEGRITY ASSESSMENT:
    Assess the organization, formatting, and data quality:
    1. Are problem entries properly formatted and structured?
    2. Do all problems have clear, specific diagnostic names?
    3. Are problem statuses clearly indicated?
    4. Is the problem list well-organized and consistent?
    5. Are there vague or unclear diagnostic terms?

    CODING STRUCTURE ASSESSMENT:
    Use your medical coding knowledge to assess:
    6. Are medical codes properly formatted for diagnoses (SNOMED CT/ICD patterns)?
    7. Do code systems follow standard diagnostic coding conventions?
    8. Are display names consistent with expected medical diagnostic terminology?
    9. Do you see obvious code-to-name mismatches or inappropriate diagnostic codes?
    10. Are codes suitable for diagnostic accuracy and clinical communication?
                """
            },
            'narrative_consistency': {
                'lab_results': """
    NARRATIVE CONSISTENCY ASSESSMENT:
    Compare narrative text with structured data:
    1. Does the narrative accurately describe the laboratory findings?
    2. Are significant lab abnormalities mentioned in the narrative?
    3. Is the narrative interpretation consistent with structured data?
    4. Are there discrepancies between narrative and lab data?
    5. Does the narrative provide appropriate clinical context?

    CODING CONSISTENCY ASSESSMENT:
    Use your medical coding knowledge to assess:
    6. Are lab test names in narrative consistent with any medical codes present?
    7. Do coded entries match their narrative descriptions?
    8. Are medical terms used consistently between narrative and structured data?
    9. Do you see conflicts between coded data and narrative clinical descriptions?
                """,
                'medications': """
    NARRATIVE CONSISTENCY ASSESSMENT:
    Compare narrative text with structured data:
    1. Does the narrative accurately describe the medication regimen?
    2. Are significant medications mentioned in the narrative?
    3. Is the narrative consistent with the structured medication data?
    4. Are there discrepancies between narrative and medication list?
    5. Does the narrative explain medication rationale appropriately?

    CODING CONSISTENCY ASSESSMENT:
    Use your medical coding knowledge to assess:
    6. Are medication names in narrative consistent with any medical codes present?
    7. Do coded medications match their narrative descriptions?
    8. Are drug names used consistently between narrative and structured data?
    9. Do you see conflicts between coded medications and narrative descriptions?
                """,
                'problems': """
    NARRATIVE CONSISTENCY ASSESSMENT:
    Compare narrative text with structured data:
    1. Does the narrative accurately describe the patient's problems?
    2. Are active diagnoses mentioned in the narrative?
    3. Is the narrative consistent with the structured problem data?
    4. Are there discrepancies between narrative and problem list?
    5. Does the narrative provide appropriate clinical context?

    CODING CONSISTENCY ASSESSMENT:
    Use your medical coding knowledge to assess:
    6. Are diagnostic terms in narrative consistent with any medical codes present?
    7. Do coded diagnoses match their narrative descriptions?
    8. Are medical terms used consistently between narrative and structured data?
    9. Do you see conflicts between coded diagnoses and narrative clinical descriptions?
                """
            }
        }

        return questions.get(dimension, {}).get(section_type,
                                                f"Please assess the {dimension.replace('_', ' ')} of this {section_type.replace('_', ' ')} section using your clinical and medical coding knowledge.")

    def _query_openai(self, question: str, section_type: str, dimension: str) -> Dict[str, Any]:
        """Query OpenAI for pure LLM analysis."""

        system_prompt = f"""You are a clinical expert specializing in {section_type.replace('_', ' ')} and {dimension.replace('_', ' ')} assessment. Use your medical training and clinical knowledge to provide accurate assessments. Do not ask for reference ranges or additional information - use your clinical expertise. Always respond in the requested JSON format."""

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=600
        )

        response_text = response.choices[0].message.content
        return self._parse_llm_response(response_text, section_type, dimension)

    def _query_openai_lab(self, question: str, lab_name: str, value: str, unit: str) -> Dict[str, Any]:
        """Query OpenAI for pure lab analysis."""

        system_prompt = """You are a clinical laboratory medicine specialist. Use your medical training to assess laboratory results. Do not ask for reference ranges - use your clinical knowledge of normal values. Always respond in the requested JSON format."""

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=400
        )

        response_text = response.choices[0].message.content
        return self._parse_lab_response(response_text, lab_name, value, unit)

    def _query_claude(self, question: str, section_type: str, dimension: str) -> Dict[str, Any]:
        """Query Claude for pure LLM analysis."""

        print(f"🤖 DEBUG: Calling Claude for {section_type} - {dimension}")
        # TEMPORARY FIX - for testing only:
        if not self.claude_client:
            print("❌ No Claude client available")
            return self._fallback_response(section_type, dimension)

        system_prompt = f"""You are a clinical expert specializing in {section_type.replace('_', ' ')} and {dimension.replace('_', ' ')} assessment. Use your medical training and clinical knowledge to provide accurate assessments. Do not ask for reference ranges or additional information - use your clinical expertise. Always respond in the requested JSON format."""

        try:
            print(f"🤖 Sending question to Claude...")
            response = self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=600,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": question}
                ]
            )

            response_text = response.content[0].text
            print(f"✅ Got response from Claude: {response_text[:100]}...")

            return self._parse_llm_response(response_text, section_type, dimension)

        except Exception as e:
            print(f"⚠️ Claude query error: {e}")
            return self._fallback_response(section_type, dimension)

    def _query_claude_lab(self, question: str, lab_name: str, value: str, unit: str) -> Dict[str, Any]:
        """Query Claude for pure lab analysis."""
        # Placeholder for Claude implementation
        return self._fallback_lab_response(lab_name, value, unit)

    def _parse_llm_response(self, response_text: str, section_type: str, dimension: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""

        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback parsing
                analysis = self._parse_text_response(response_text)

            # Validate and clean response
            return self._validate_response(analysis, section_type, dimension, response_text)

        except Exception as e:
            print(f"⚠️ Error parsing LLM response: {e}")
            return self._fallback_response(section_type, dimension)

    def _parse_lab_response(self, response_text: str, lab_name: str, value: str, unit: str) -> Dict[str, Any]:
        """Parse LLM lab response."""

        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback parsing
                analysis = self._parse_lab_text_response(response_text)

            # Validate and clean response
            return self._validate_lab_response(analysis, lab_name, value, unit, response_text)

        except Exception as e:
            print(f"⚠️ Error parsing lab LLM response: {e}")
            return self._fallback_lab_response(lab_name, value, unit)

    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback text parsing."""

        text_lower = response_text.lower()

        if any(word in text_lower for word in ['excellent', 'outstanding']):
            assessment, score = 'EXCELLENT', 90
        elif any(word in text_lower for word in ['good', 'adequate']):
            assessment, score = 'GOOD', 75
        elif any(word in text_lower for word in ['fair', 'moderate']):
            assessment, score = 'FAIR', 60
        elif any(word in text_lower for word in ['poor', 'inadequate']):
            assessment, score = 'POOR', 40
        elif any(word in text_lower for word in ['critical', 'dangerous']):
            assessment, score = 'CRITICAL', 20
        else:
            assessment, score = 'FAIR', 50

        return {
            'overall_assessment': assessment,
            'score': score,
            'clinical_findings': ['Text analysis completed'],
            'recommendations': ['Review recommended'],
            'specific_issues': ['Text parsing used'],
            'confidence': 60,
            'rationale': 'Fallback text parsing'
        }

    def _parse_lab_text_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback lab text parsing."""

        text_lower = response_text.lower()

        if any(word in text_lower for word in ['critical', 'emergency', 'urgent']):
            status, risk_level, score = 'CRITICAL', 'CRITICAL', 90
        elif any(word in text_lower for word in ['abnormal', 'elevated', 'low', 'high']):
            status, risk_level, score = 'ABNORMAL', 'MODERATE', 70
        else:
            status, risk_level, score = 'NORMAL', 'LOW', 20

        return {
            'status': status,
            'risk_level': risk_level,
            'clinical_significance': 'Text analysis completed',
            'immediate_actions': ['Clinical review recommended'],
            'severity_score': score,
            'confidence': 60
        }

    def _validate_response(self, analysis: Dict, section_type: str, dimension: str, original_response: str) -> Dict[
        str, Any]:
        """Validate LLM response."""

        # Set defaults
        defaults = {
            'overall_assessment': 'FAIR',
            'score': 50,
            'clinical_findings': ['Analysis completed'],
            'recommendations': ['Review recommended'],
            'specific_issues': ['Standard assessment'],
            'confidence': 70,
            'rationale': 'Standard assessment'
        }

        for key, default_value in defaults.items():
            if key not in analysis or not analysis[key]:
                analysis[key] = default_value

        # Validate assessment
        valid_assessments = ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL']
        if analysis['overall_assessment'] not in valid_assessments:
            analysis['overall_assessment'] = 'FAIR'

        # Ensure valid ranges
        analysis['score'] = max(0, min(100, int(analysis.get('score', 50))))
        analysis['confidence'] = max(0, min(100, int(analysis.get('confidence', 70))))

        # Ensure lists
        for list_field in ['clinical_findings', 'recommendations', 'specific_issues']:
            if not isinstance(analysis[list_field], list):
                analysis[list_field] = [str(analysis[list_field])] if analysis[list_field] else []

        # Add metadata
        analysis['section_type'] = section_type
        analysis['dimension'] = dimension
        analysis['original_response'] = original_response[:200]

        return analysis

    def _validate_lab_response(self, analysis: Dict, lab_name: str, value: str, unit: str, original_response: str) -> \
    Dict[str, Any]:
        """Validate lab LLM response."""

        # Set defaults
        defaults = {
            'status': 'ABNORMAL',
            'risk_level': 'MODERATE',
            'clinical_significance': 'Requires evaluation',
            'immediate_actions': ['Clinical review recommended'],
            'severity_score': 50,
            'confidence': 70
        }

        for key, default_value in defaults.items():
            if key not in analysis or not analysis[key]:
                analysis[key] = default_value

        # Validate status
        valid_statuses = ['NORMAL', 'ABNORMAL', 'CRITICAL']
        if analysis['status'] not in valid_statuses:
            analysis['status'] = 'ABNORMAL'

        # Validate risk level
        valid_risks = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        if analysis['risk_level'] not in valid_risks:
            analysis['risk_level'] = 'MODERATE'

        # Ensure valid ranges
        analysis['severity_score'] = max(0, min(100, int(analysis.get('severity_score', 50))))
        analysis['confidence'] = max(0, min(100, int(analysis.get('confidence', 70))))

        # Add metadata
        analysis['lab_name'] = lab_name
        analysis['value'] = value
        analysis['unit'] = unit
        analysis['original_response'] = original_response[:200]

        return analysis

    def _fallback_response(self, section_type: str, dimension: str) -> Dict[str, Any]:
        """Fallback when LLM unavailable."""

        return {
            'overall_assessment': 'FAIR',
            'score': 50,
            'clinical_findings': [f'LLM analysis unavailable for {section_type} {dimension}'],
            'recommendations': ['Manual clinical assessment needed'],
            'specific_issues': ['LLM service not available'],
            'confidence': 0,
            'rationale': 'Fallback - LLM not available',
            'section_type': section_type,
            'dimension': dimension,
            'original_response': 'Fallback response'
        }

    def _fallback_lab_response(self, lab_name: str, value: str, unit: str) -> Dict[str, Any]:
        """Fallback when LLM unavailable for labs."""

        return {
            'status': 'UNKNOWN',
            'risk_level': 'MODERATE',
            'clinical_significance': 'LLM analysis unavailable',
            'immediate_actions': ['Manual clinical assessment needed'],
            'severity_score': 50,
            'confidence': 0,
            'lab_name': lab_name,
            'value': value,
            'unit': unit,
            'original_response': 'Fallback - LLM not available'
        }


"""
Part 2: Main Clinical Analyzer ( LLM - No Knowledge Base)
============================================================

This part contains the AIPoweredClinicalAnalyzer class that orchestrates
all clinical analysis using ONLY LLM reasoning - no knowledge base at all.
"""


class AIPoweredClinicalAnalyzer:
    """
    Pure LLM Clinical Analyzer - ZERO knowledge base dependency.

    Orchestrates comprehensive clinical analysis using ONLY direct LLM
    communication for ALL assessments including drug interactions,
    clinical plausibility, and quality dimensions.
    """

    def __init__(self, device='cpu'):
        """
        Initialize pure LLM clinical analyzer.

        Args:
            device: Not used in pure LLM mode (legacy parameter)
        """
        self.device = device
        self.model_type = "Pure LLM Analysis"

        # NO knowledge base initialization
        # NO reference ranges
        # NO drug interaction databases
        # NO clinical rules

        print(" Initializing Pure LLM Clinical Analyzer - NO knowledge base")

        # Initialize ONLY the LLM analyzer
        # self.generative_analyzer = GenerativeClinicalAnalyzer(
        #     provider="openai",
        #     api_key=os.getenv('OPENAI_API_KEY')
        self.generative_analyzer = GenerativeClinicalAnalyzer(
            provider="claude",  # ← Use Claude instead!
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )

        print("✅ LLM mode - All analysis via AI reasoning")

    # ========================================================================
    # PRIMARY ANALYSIS METHODS - PURE LLM ONLY
    # ========================================================================

    def analyze_clinical_plausibility(self, structured_data: List[Dict], narrative_text: str,
                                      section_type: str, patient_age: int = None) -> Dict[str, Any]:
        """
        Pure LLM clinical plausibility analysis - NO knowledge base.

        Uses ONLY LLM reasoning for:
        - Drug interaction detection
        - Clinical value assessment
        - Age-appropriate analysis
        - Risk stratification

        Args:
            structured_data: Clinical data items
            narrative_text: Narrative text
            section_type: 'lab_results', 'medications', 'problems'
            patient_age: Patient age for context

        Returns:
            Dict with LLM analysis results
        """
        if not structured_data:
            return {
                'score': 0.0,
                'details': {
                    'checks_performed': ['Data Availability Check'],
                    'findings': ['No structured data available for analysis'],
                    'recommendations': ['Ensure structured data is present in the document'],
                    'ai_model_used': self.model_type,
                    'analysis_type': 'Pure LLM Clinical Analysis'
                }
            }

        print(f" Starting LLM clinical plausibility analysis for {section_type}")

        try:
            #  LLM analysis - NO knowledge base consulted
            llm_analysis = self.generative_analyzer.analyze_section_comprehensive(
                structured_data, narrative_text, section_type, 'clinical_plausibility', patient_age
            )

            # Convert LLM analysis to standard format
            score = llm_analysis.get('score', 50.0)
            findings = llm_analysis.get('clinical_findings', [])
            recommendations = llm_analysis.get('recommendations', [])

            # Format findings with Pure LLM indicators
            formatted_findings = []
            for finding in findings:
                formatted_findings.append(f" LLM: {finding}")

            # Format recommendations with Pure LLM indicators
            formatted_recommendations = []
            for rec in recommendations:
                formatted_recommendations.append(f" LLM EXPERT: {rec}")

            # Add LLM assessment summary
            assessment = llm_analysis.get('overall_assessment', 'FAIR')
            confidence = llm_analysis.get('confidence', 70)
            rationale = llm_analysis.get('rationale', 'Pure LLM clinical assessment')

            formatted_findings.append(f"🎯 LLM CLINICAL ASSESSMENT: {assessment} (Confidence: {confidence}%)")
            formatted_findings.append(f"🧠 LLM REASONING: {rationale}")

        except Exception as e:
            print(f"⚠️ Error in Pure LLM clinical analysis: {e}")
            score = 75.0
            formatted_findings = [f"⚠️ Error in LLM analysis: Clinical review recommended"]
            formatted_recommendations = ["🔧 Manual clinical review recommended"]

        return {
            'score': float(max(0.0, min(100.0, score))),
            'details': {
                'checks_performed': [' LLM Clinical Plausibility Analysis'],
                'findings': formatted_findings,
                'recommendations': formatted_recommendations,
                'ai_model_used': self.model_type,
                'analysis_type': 'Pure LLM Clinical Analysis'
            }
        }

    def analyze_clinical_completeness(self, structured_data: List[Dict], section_type: str,
                                      patient_age: int = None) -> Dict[str, Any]:
        """
        Pure LLM completeness analysis - NO knowledge base.

        Uses ONLY LLM to assess:
        - Data comprehensiveness
        - Missing clinical elements
        - Age-appropriate completeness

        Args:
            structured_data: Clinical data items
            section_type: Section being analyzed
            patient_age: Patient age for context

        Returns:
            Dict with LLM completeness analysis
        """
        if not structured_data:
            return {
                'score': 0.0,
                'findings': ['No structured data available for completeness analysis'],
                'recommendations': ['Add structured clinical data to the section']
            }

        print(f"🤖 Starting LLM completeness analysis for {section_type}")

        try:
            # LLM completeness analysis
            llm_analysis = self.generative_analyzer.analyze_section_comprehensive(
                structured_data, "", section_type, 'completeness', patient_age
            )

            # Convert LLM analysis to standard format
            score = llm_analysis.get('score', 50.0)
            findings = llm_analysis.get('clinical_findings', [])
            recommendations = llm_analysis.get('recommendations', [])

            # Format findings with Pure LLM indicators
            formatted_findings = []
            for finding in findings:
                formatted_findings.append(f" LLM COMPLETENESS: {finding}")

            # Format recommendations
            formatted_recommendations = []
            for rec in recommendations:
                formatted_recommendations.append(f" LLM GUIDANCE: {rec}")

            # Add assessment summary
            assessment = llm_analysis.get('overall_assessment', 'FAIR')
            confidence = llm_analysis.get('confidence', 70)

            formatted_findings.append(f"📊 LLM COMPLETENESS SCORE: {assessment} ({score:.1f}/100)")
            formatted_findings.append(f"🎯 LLM CONFIDENCE: {confidence}%")

        except Exception as e:
            print(f"⚠️ Error in Pure LLM completeness analysis: {e}")
            score = 50.0
            formatted_findings = [f"⚠️ Error in LLM completeness analysis"]
            formatted_recommendations = ["🔧 Manual completeness review recommended"]

        return {
            'score': float(max(0.0, min(100.0, score))),
            'findings': formatted_findings,
            'recommendations': formatted_recommendations
        }

    def analyze_structural_coherence(self, structured_data: List[Dict], section_type: str) -> Dict[str, Any]:
        """
        Pure LLM structural analysis - NO knowledge base.

        Uses ONLY LLM to assess:
        - Data formatting and organization
        - Structural integrity
        - Consistency patterns

        Args:
            structured_data: Clinical data items
            section_type: Section being analyzed

        Returns:
            Dict with LLM structural analysis
        """
        if not structured_data:
            return {
                'score': 0.0,
                'findings': ['No structured data available for coherence analysis'],
                'recommendations': ['Add structured data to enable coherence analysis']
            }

        print(f"🤖 Starting Pure LLM structural analysis for {section_type}")

        try:
            # Pure LLM structural analysis
            llm_analysis = self.generative_analyzer.analyze_section_comprehensive(
                structured_data, "", section_type, 'structural_integrity'
            )

            # Convert LLM analysis to standard format
            score = llm_analysis.get('score', 50.0)
            findings = llm_analysis.get('clinical_findings', [])
            recommendations = llm_analysis.get('recommendations', [])

            # Format findings with Pure LLM indicators
            formatted_findings = []
            for finding in findings:
                formatted_findings.append(f"🤖 LLM STRUCTURE: {finding}")

            # Format recommendations
            formatted_recommendations = []
            for rec in recommendations:
                formatted_recommendations.append(f"🤖 LLM STRUCTURAL FIX: {rec}")

            # Add assessment summary
            assessment = llm_analysis.get('overall_assessment', 'FAIR')
            confidence = llm_analysis.get('confidence', 70)

            formatted_findings.append(f"🏗️ LLM STRUCTURAL SCORE: {assessment} ({score:.1f}/100)")
            formatted_findings.append(f"🎯 LLM CONFIDENCE: {confidence}%")

        except Exception as e:
            print(f"⚠️ Error in Pure LLM structural analysis: {e}")
            score = 50.0
            formatted_findings = [f"⚠️ Error in LLM structural analysis"]
            formatted_recommendations = ["🔧 Manual structural review recommended"]

        return {
            'score': float(max(0.0, min(100.0, score))),
            'findings': formatted_findings,
            'recommendations': formatted_recommendations
        }

    # ========================================================================
    # INDIVIDUAL ITEM ANALYSIS - PURE LLM ONLY
    # ========================================================================

    def _ai_analyze_lab_results(self, lab_data: List[Dict], patient_age: int, checks: List[str],
                                findings: List[str], recommendations: List[str]) -> Dict[str, float]:
        """Pure LLM lab analysis - NO reference ranges."""
        score = 100.0
        checks.append("🤖 Pure LLM Laboratory Analysis")

        if not lab_data:
            findings.append("⚠️ No laboratory data found for analysis")
            return {'score': 0}

        print(f"🤖 Pure LLM analyzing {len(lab_data)} lab results...")

        total_penalty = 0

        for i, lab in enumerate(lab_data):
            lab_name = lab.get('display_name', '').lower().strip()
            value_str = lab.get('value', '').strip()
            unit = lab.get('unit', '').strip()

            if not lab_name or not value_str:
                findings.append(f"⚠️ Incomplete lab data for entry {i + 1}")
                total_penalty += 5
                continue

            print(f"🤖 Pure LLM analyzing: {lab_name} = {value_str} {unit}")

            # Pure LLM analysis - NO knowledge base lookup
            llm_analysis = self.generative_analyzer.analyze_lab_direct(
                lab_name, value_str, unit, patient_age
            )

            # Convert to scoring format
            assessment = self._convert_pure_llm_analysis_to_score(llm_analysis, lab_name, value_str, unit)

            total_penalty += assessment['penalty']
            findings.extend(assessment['findings'])
            recommendations.extend(assessment['recommendations'])

        final_score = max(0, score - total_penalty)
        print(f"🤖 Pure LLM Lab Analysis Complete - Final Score: {final_score:.1f}")

        return {'score': final_score}

    def _ai_analyze_medications(self, med_data: List[Dict], patient_age: int, checks: List[str],
                                findings: List[str], recommendations: List[str]) -> Dict[str, float]:
        """Pure LLM medication analysis - NO drug interaction database."""
        score = 100.0
        checks.append(" Pure LLM Medication & Drug Interaction Analysis")

        if not med_data:
            findings.append("⚠️ No medication data found for analysis")
            return {'score': 0}

        print(f" LLM analyzing {len(med_data)} medications for interactions...")

        try:
            # Pure LLM medication and drug interaction analysis
            llm_analysis = self.generative_analyzer.analyze_section_comprehensive(
                med_data, "", 'medications', 'clinical_plausibility', patient_age
            )

            # Convert to scoring format
            score = llm_analysis.get('score', 50.0)
            llm_findings = llm_analysis.get('clinical_findings', [])
            llm_recommendations = llm_analysis.get('recommendations', [])

            # Add LLM findings and recommendations
            for finding in llm_findings:
                findings.append(f" LLM MEDICATION: {finding}")

            for rec in llm_recommendations:
                recommendations.append(f" LLM DRUG SAFETY: {rec}")

            # Add assessment summary
            assessment = llm_analysis.get('overall_assessment', 'FAIR')
            confidence = llm_analysis.get('confidence', 70)
            findings.append(f"💊 LLM MEDICATION ASSESSMENT: {assessment} (Confidence: {confidence}%)")

        except Exception as e:
            print(f"⚠️ Error in LLM medication analysis: {e}")
            score = 50.0
            findings.append("⚠️ LLM medication analysis error - manual review recommended")

        final_score = max(0, score)
        print(f" LLM Medication Analysis Complete - Final Score: {final_score:.1f}")

        return {'score': final_score}

    def _ai_analyze_problems(self, problem_data: List[Dict], patient_age: int, checks: List[str],
                             findings: List[str], recommendations: List[str]) -> Dict[str, float]:
        """Pure LLM problem analysis - NO diagnostic rules."""
        score = 100.0
        checks.append(" LLM Problem & Diagnosis Analysis")

        if not problem_data:
            findings.append("⚠️ No problem data found for analysis")
            return {'score': 0}

        print(f" LLM analyzing {len(problem_data)} problems...")

        try:
            # Pure LLM problem analysis
            llm_analysis = self.generative_analyzer.analyze_section_comprehensive(
                problem_data, "", 'problems', 'clinical_plausibility', patient_age
            )

            # Convert to scoring format
            score = llm_analysis.get('score', 50.0)
            llm_findings = llm_analysis.get('clinical_findings', [])
            llm_recommendations = llm_analysis.get('recommendations', [])

            # Add LLM findings and recommendations
            for finding in llm_findings:
                findings.append(f" LLM DIAGNOSIS: {finding}")

            for rec in llm_recommendations:
                recommendations.append(f" LLM DIAGNOSTIC: {rec}")

            # Add assessment summary
            assessment = llm_analysis.get('overall_assessment', 'FAIR')
            confidence = llm_analysis.get('confidence', 70)
            findings.append(f"🏥 LLM DIAGNOSTIC ASSESSMENT: {assessment} (Confidence: {confidence}%)")

        except Exception as e:
            print(f"⚠️ Error in Pure LLM problem analysis: {e}")
            score = 50.0
            findings.append("⚠️ Pure LLM problem analysis error - manual review recommended")

        final_score = max(0, score)
        print(f" LLM Problem Analysis Complete - Final Score: {final_score:.1f}")

        return {'score': final_score}

    # ========================================================================
    # UTILITY METHODS - NO KNOWLEDGE BASE
    # ========================================================================

    def _convert_pure_llm_analysis_to_score(self, llm_analysis: Dict, lab_name: str, value_str: str, unit: str) -> Dict:
        """Convert Pure LLM analysis to scoring format - NO reference ranges used."""

        status = llm_analysis.get('status', 'UNKNOWN')
        risk_level = llm_analysis.get('risk_level', 'MODERATE')
        clinical_significance = llm_analysis.get('clinical_significance', '')
        immediate_actions = llm_analysis.get('immediate_actions', [])
        severity_score = llm_analysis.get('severity_score', 50)
        confidence = llm_analysis.get('confidence', 70)

        findings = []
        recommendations = []
        penalty = 0

        # Convert LLM assessment to findings and penalties
        if status == 'CRITICAL':
            penalty = 25
            findings.append(f"🔴 CRITICAL: {lab_name.title()} = {value_str} {unit}")
            findings.append(f"🤖 LLM CLINICAL REASONING: {clinical_significance}")
            findings.append(f"🎯 LLM CONFIDENCE: {confidence}%")

            for action in immediate_actions:
                recommendations.append(f"🚨 LLM URGENT: {action}")

        elif status == 'ABNORMAL':
            if risk_level == 'HIGH':
                penalty = 15
            else:
                penalty = 10
            findings.append(f"⚠️ ABNORMAL: {lab_name.title()} = {value_str} {unit}")
            findings.append(f"🤖 LLM CLINICAL REASONING: {clinical_significance}")
            findings.append(f"🎯 LLM CONFIDENCE: {confidence}%")

            for action in immediate_actions:
                recommendations.append(f"🔍 LLM RECOMMENDATION: {action}")

        elif status == 'NORMAL':
            penalty = 0
            findings.append(f"✅ NORMAL: {lab_name.title()} = {value_str} {unit}")
            findings.append(f"🤖 LLM CLINICAL REASONING: {clinical_significance}")
            findings.append(f"🎯 LLM CONFIDENCE: {confidence}%")

            for action in immediate_actions:
                recommendations.append(f"📋 LLM GUIDANCE: {action}")

        else:  # UNKNOWN
            penalty = 5
            findings.append(f"❓ UNKNOWN: {lab_name.title()} = {value_str} {unit}")
            findings.append(f"🤖 LLM: Unable to determine clinical status")
            recommendations.append(f"🔧 Manual clinical review required")

        # Add risk level information from LLM
        findings.append(f"⚖️ LLM RISK ASSESSMENT: {risk_level} (Severity: {severity_score}/100)")

        return {
            'penalty': penalty,
            'findings': findings,
            'recommendations': recommendations,
            'llm_analysis': llm_analysis
        }

    def _extract_numeric_value(self, value_str: str) -> float:
        """Extract numeric value from string - utility method."""
        if not value_str:
            return None

        # Clean the string and extract numeric value
        clean_value = re.sub(r'[<>=]+', '', str(value_str))
        numeric_matches = re.findall(r'-?\d+\.?\d*', clean_value)

        if numeric_matches:
            try:
                return float(numeric_matches[0])
            except ValueError:
                pass
        return None
"""


Part 3: CCDA Document Processor (LLM)
==========================================

This part contains the CCDASectionScorer class that processes CCDA documents
and coordinates Pure LLM analysis across all sections and dimensions.
"""


class CCDASectionScorer:
    """
    CCDA Section Quality Scorer with LLM Analysis.

    Main orchestrator that processes CCDA XML documents and coordinates
    comprehensive quality analysis using direct LLM reasoning.
    NO knowledge base, NO hardcoded rules, NO reference ranges.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LLM CCDA Section Quality Scorer.

        Args:
            config: Configuration dictionary for customization
        """
        default_config = {
            "device": "cpu",  # Not used in LLM mode
            "use_ai_analysis": True,
            "namespaces": {
                "cda": "urn:hl7-org:v3",
                "sdtc": "urn:hl7-org:sdtc"
            },
            "thresholds": {
                "high_quality": 0.8,
                "medium_quality": 0.6
            }
        }

        self.config = default_config
        if config:
            self._update_config_recursive(self.config, config)

        # Initialize Pure LLM Clinical Analyzer for ALL dimensions
        if self.config.get('use_ai_analysis', True):
            print(" Initializing LLM Analysis for ALL sections and ALL dimensions...")
            self.clinical_analyzer = AIPoweredClinicalAnalyzer(device=self.config['device'])
        else:
            print("📋 LLM analysis disabled - basic mode only...")
            self.clinical_analyzer = None

    def _add_code_metrics(self, section_result: Dict[str, Any], structured_data: List[Dict], section_type: str) -> Dict[
        str, Any]:
        """Add simple code metrics to section results for LLM analysis insight."""

        if not structured_data:
            return section_result

        # Simple counts - NO validation rules
        total_items = len(structured_data)
        items_with_codes = sum(1 for item in structured_data if item.get('code'))
        items_with_code_systems = sum(1 for item in structured_data if item.get('code_system'))
        items_with_display_names = sum(1 for item in structured_data if item.get('display_name'))

        # Basic metrics for LLM context
        code_metrics = {
            'total_items': total_items,
            'items_with_codes': items_with_codes,
            'items_with_code_systems': items_with_code_systems,
            'items_with_display_names': items_with_display_names,
            'code_presence_percentage': round((items_with_codes / total_items) * 100, 1) if total_items > 0 else 0
        }

        # Add to section result (for reporting only)
        section_result['code_metrics'] = code_metrics

        return section_result

    def _update_config_recursive(self, config: Dict, updates: Dict):
        """Recursively update configuration."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                self._update_config_recursive(config[key], value)
            else:
                config[key] = value

    # ========================================================================
    # MAIN DOCUMENT SCORING - LLM
    # ========================================================================

    def score_ccda(self, ccda_file_path: str) -> Dict[str, Any]:
        """
        Score a CCDA document with LLM analysis for ALL dimensions.

        Args:
            ccda_file_path: Path to CCDA XML file

        Returns:
            Dict with comprehensive LLM scoring results
        """
        try:
            # Parse XML document
            parser = etree.XMLParser(strip_cdata=False, recover=True)
            doc = etree.parse(ccda_file_path, parser)

            # Extract document metadata
            document_id = self._extract_document_id(doc)
            patient_id = self._extract_patient_id(doc)
            patient_demographics = self.extract_patient_demographics(doc)
            document_metadata = self._extract_document_metadata(doc)

            print(f"📄 Processing document ID: {document_id}, Patient ID: {patient_id}")
            if patient_demographics.get('age'):
                print(
                    f"👤 Patient: Age {patient_demographics['age']}, Gender: {patient_demographics.get('gender', 'Unknown')}")

            # Score individual sections with LLM analysis
            sections = {}
            sections['lab_results'] = self.score_lab_results(doc)
            sections['medications'] = self.score_medications(doc)
            sections['problems'] = self.score_problems(doc)

            # Calculate overall score
            section_scores = []
            for section_data in sections.values():
                if section_data.get('section_present', False):
                    section_scores.append(section_data.get('overall_score', 0))

            overall_score = sum(section_scores) / len(section_scores) if section_scores else 0

            # Determine quality level
            if overall_score >= 80:
                quality_level = "High"
            elif overall_score >= 60:
                quality_level = "Medium"
            else:
                quality_level = "Low"

            return {
                'document_id': document_id,
                'patient_id': patient_id,
                'timestamp': datetime.now().isoformat(),
                'overall_score': overall_score,
                'quality_level': quality_level,
                'sections': sections,
                'patient_demographics': patient_demographics,
                'document_metadata': document_metadata,
                'analysis_config': {
                    'ai_enabled': self.config.get('use_ai_analysis', True),
                    'analyzer_type': ' LLM Analysis - NO Knowledge Base',
                    'model_type': 'Direct LLM Reasoning'
                }
            }

        except Exception as e:
            print(f"Error scoring CCDA document: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'overall_score': 0,
                'quality_level': 'Error',
                'document_id': 'error',
                'patient_id': 'error',
                'timestamp': datetime.now().isoformat(),
                'sections': {},
                'patient_demographics': {},
                'document_metadata': {}
            }

    # ========================================================================
    # SECTION SCORING METHODS -  LLM
    # ========================================================================

    def score_lab_results(self, doc: etree._Element) -> Dict[str, Any]:
        """Score Lab Results section using PURE LLM analysis for ALL dimensions."""
        print('🔬 PURE LLM Scoring Lab Results section - ALL DIMENSIONS...')

        try:
            # Extract lab section data
            structured_data = self.extract_structured_data(doc, 'lab_results')
            narrative_text = self.extract_narrative_text(doc, 'lab_results')

            print("🔍 DEBUG - Actual lab data extracted:")
            for lab in structured_data:
                name = lab.get('display_name', 'Unknown')
                value = lab.get('value', 'Unknown')
                unit = lab.get('unit', '')
                print(f"  {name}: '{value}' {unit}")

            if not structured_data and not narrative_text:
                return self._create_default_section_score('lab_results', False)

            # Extract patient demographics for LLM analysis
            demographics = self.extract_patient_demographics(doc)
            patient_age = demographics.get('age')

            print(f"🔬 Found {len(structured_data)} lab results, patient age: {patient_age}")

            # PURE LLM ANALYSIS FOR ALL DIMENSIONS - NO KNOWLEDGE BASE
            scores, issues, recommendations = self._pure_llm_score_section(
                structured_data, narrative_text, 'lab_results', patient_age
            )

            # Compile results
            section_result = self._compile_section_score(
                scores, issues, recommendations, 'lab_results', True

            )
            section_result = self._add_code_metrics(section_result, structured_data, 'lab_results')

            section_result['clinical_details'] = {
                'ai_model_used': 'Pure LLM - NO Knowledge Base',
                'analysis_type': 'Pure LLM Clinical Analysis'
            }

            print(f" LLM Analysis Complete - Overall Score: {section_result['overall_score']:.1f}")
            return section_result

        except Exception as e:
            print(f"Error scoring lab results: {e}")
            return self._create_error_section_score('lab_results', str(e))

    def score_medications(self, doc: etree._Element) -> Dict[str, Any]:
        """Score Medications section using PURE LLM analysis for ALL dimensions."""
        print('💊 PURE LLM Scoring Medications section - ALL DIMENSIONS...')

        try:
            # Extract medication section data
            structured_data = self.extract_structured_data(doc, 'medications')
            narrative_text = self.extract_narrative_text(doc, 'medications')

            if not structured_data and not narrative_text:
                return self._create_default_section_score('medications', False)

            # Extract patient demographics for LLM analysis
            demographics = self.extract_patient_demographics(doc)
            patient_age = demographics.get('age')

            print(f"💊 Found {len(structured_data)} medications, patient age: {patient_age}")

            # PURE LLM ANALYSIS FOR ALL DIMENSIONS - NO DRUG INTERACTION DATABASE
            scores, issues, recommendations = self._pure_llm_score_section(
                structured_data, narrative_text, 'medications', patient_age
            )

            # Compile results
            section_result = self._compile_section_score(
                scores, issues, recommendations, 'medications', True
            )
            section_result = self._add_code_metrics(section_result, structured_data, 'medications')
            section_result['clinical_details'] = {
                'ai_model_used': 'Pure LLM - NO Knowledge Base',
                'analysis_type': 'Pure LLM Clinical Analysis'
            }

            print(f"🤖 Medication LLM Analysis Complete - Overall Score: {section_result['overall_score']:.1f}")
            return section_result

        except Exception as e:
            print(f"Error scoring medications: {e}")
            return self._create_error_section_score('medications', str(e))

    def score_problems(self, doc: etree._Element) -> Dict[str, Any]:
        """Score Problems section using PURE LLM analysis for ALL dimensions."""
        print('🏥 PURE LLM Scoring Problems section - ALL DIMENSIONS...')

        try:
            # Extract problems section data
            structured_data = self.extract_structured_data(doc, 'problems')
            narrative_text = self.extract_narrative_text(doc, 'problems')

            if not structured_data and not narrative_text:
                return self._create_default_section_score('problems', False)

            # Extract patient demographics for LLM analysis
            demographics = self.extract_patient_demographics(doc)
            patient_age = demographics.get('age')

            print(f"🏥 Found {len(structured_data)} problems, patient age: {patient_age}")

            # PURE LLM ANALYSIS FOR ALL DIMENSIONS - NO DIAGNOSTIC RULES
            scores, issues, recommendations = self._pure_llm_score_section(
                structured_data, narrative_text, 'problems', patient_age
            )

            # Compile results
            section_result = self._compile_section_score(
                scores, issues, recommendations, 'problems', True
            )
            section_result = self._add_code_metrics(section_result, structured_data, 'problems')

            section_result['clinical_details'] = {
                'ai_model_used': 'Pure LLM - NO Knowledge Base',
                'analysis_type': 'Pure LLM Clinical Analysis'
            }

            print(f" Problems LLM Analysis Complete - Overall Score: {section_result['overall_score']:.1f}")
            return section_result

        except Exception as e:
            print(f"Error scoring problems: {e}")
            return self._create_error_section_score('problems', str(e))

    def _pure_llm_score_section(self, structured_data: List[Dict], narrative_text: str,
                                section_type: str, patient_age: int = None) -> tuple:
        """PURE LLM scoring for ALL dimensions of ANY section - NO knowledge base."""
        scores = {
            'completeness': 0.0,
            'structural_integrity': 0.0,
            'clinical_plausibility': 0.0,
            'narrative_consistency': 0.0
        }

        issues = []
        recommendations = []

        print(f" Starting LLM analysis for ALL dimensions of {section_type}")

        if self.clinical_analyzer:
            try:
                # 1.  LLM COMPLETENESS ANALYSIS
                completeness_analysis = self.clinical_analyzer.analyze_clinical_completeness(
                    structured_data, section_type, patient_age
                )
                scores['completeness'] = completeness_analysis['score']
                issues.extend(completeness_analysis.get('findings', []))
                recommendations.extend(completeness_analysis.get('recommendations', []))
                print(f"✅  LLM Completeness Score: {scores['completeness']:.1f}")

                # 2.  LLM STRUCTURAL INTEGRITY ANALYSIS
                structural_analysis = self.clinical_analyzer.analyze_structural_coherence(
                    structured_data, section_type
                )
                scores['structural_integrity'] = structural_analysis['score']
                issues.extend(structural_analysis.get('findings', []))
                recommendations.extend(structural_analysis.get('recommendations', []))
                print(f"✅ LLM Structural Score: {scores['structural_integrity']:.1f}")

                # 3.  LLM CLINICAL PLAUSIBILITY ANALYSIS
                clinical_analysis = self.clinical_analyzer.analyze_clinical_plausibility(
                    structured_data, narrative_text, section_type, patient_age
                )
                scores['clinical_plausibility'] = clinical_analysis['score']
                clinical_details = clinical_analysis.get('details', {})
                issues.extend(clinical_details.get('findings', []))
                recommendations.extend(clinical_details.get('recommendations', []))
                print(f"✅ LLM Clinical Score: {scores['clinical_plausibility']:.1f}")

                # 4.  LLM NARRATIVE CONSISTENCY ANALYSIS
                narrative_analysis = self.clinical_analyzer.generative_analyzer.analyze_section_comprehensive(
                    structured_data, narrative_text, section_type, 'narrative_consistency', patient_age
                )
                scores['narrative_consistency'] = narrative_analysis.get('score', 50.0)

                # Format narrative findings
                narrative_findings = narrative_analysis.get('clinical_findings', [])
                for finding in narrative_findings:
                    issues.append(f" LLM NARRATIVE: {finding}")

                narrative_recs = narrative_analysis.get('recommendations', [])
                for rec in narrative_recs:
                    recommendations.append(f" LLM NARRATIVE FIX: {rec}")

                print(f"✅ LLM Narrative Score: {scores['narrative_consistency']:.1f}")

            except Exception as e:
                print(f"⚠️ Error in PURE LLM analysis: {e}")
                # Fallback to default scores
                for dimension in scores.keys():
                    scores[dimension] = 50.0
                issues.append(f"⚠️ Error in Pure LLM analysis: {str(e)}")
                recommendations.append("🔧 Manual review recommended due to LLM error")

        else:
            # No LLM analyzer available
            for dimension in scores.keys():
                scores[dimension] = 25.0
            issues.append("⚠️ LLM analyzer not available")
            recommendations.append("🔧 Enable LLM analysis for comprehensive assessment")

        return scores, issues, recommendations

    # ========================================================================
    # DATA EXTRACTION METHODS - XML PARSING ONLY
    # ========================================================================

    def extract_structured_data(self, doc: etree._Element, section_type: str) -> List[Dict[str, Any]]:
        """Extract structured data from CCDA section with comprehensive parsing."""
        structured_data = []
        ns = self.config["namespaces"]

        try:
            if section_type == 'lab_results':
                print("🔬 Extracting lab results...")
                structured_data = self._extract_lab_results(doc, ns)
            elif section_type == 'medications':
                print("💊 Extracting medications...")
                structured_data = self._extract_medications(doc, ns)
            elif section_type == 'problems':
                print("🏥 Extracting problems...")
                structured_data = self._extract_problems(doc, ns)

        except Exception as e:
            print(f"Error extracting structured data for {section_type}: {e}")

        print(f"✅ Extracted {len(structured_data)} {section_type} items")
        return structured_data

    def _extract_lab_results(self, doc: etree._Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract lab results with multiple parsing strategies."""
        structured_data = []

        # Multiple XPath patterns to catch different CCDA lab result structures
        lab_xpath_patterns = [
            '//cda:section[cda:templateId/@root="2.16.840.1.113883.10.20.22.2.3.1"]//cda:entry',
            '//cda:section[cda:code/@code="30954-2"]//cda:entry',
            '//cda:section[contains(translate(cda:title/text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "result")]//cda:entry',
            '//cda:entry[cda:observation[cda:value[@xsi:type="PQ" or @xsi:type="INT" or @xsi:type="REAL"]]]',
        ]

        lab_entries = []
        for pattern in lab_xpath_patterns:
            try:
                entries = doc.xpath(pattern, namespaces=ns)
                lab_entries.extend(entries)
            except Exception:
                continue

        # Remove duplicates and process
        unique_entries = self._remove_duplicate_entries(lab_entries)

        for i, entry in enumerate(unique_entries):
            try:
                lab_data = self._extract_single_lab_result(entry, ns, i)
                if lab_data and lab_data.get('display_name') and lab_data.get('value'):
                    print(f"🔍 EXTRACTED LAB: {lab_data['display_name']} = '{lab_data['value']}' {lab_data.get('unit', '')}")
                    structured_data.append(lab_data)
            except Exception as e:
                print(f"⚠️ Error extracting lab entry {i}: {e}")

        return structured_data

    def _extract_single_lab_result(self, entry: etree._Element, ns: Dict[str, str], entry_index: int) -> Dict[str, Any]:
        """Extract a single lab result with code information for LLM analysis."""
        lab_data = {
            'display_name': None,
            'value': None,
            'unit': None,
            'code': None,  # ← For LLM analysis
            'code_system': None,  # ← For LLM analysis
            'code_system_name': None,  # ← For LLM analysis
            'reference_range': None,
            'status': None,
            'type': 'lab_result',
            'entry_index': entry_index
        }

        try:

            display_name_patterns = [
                './/cda:observation/cda:code/@displayName',
                './/cda:code/@displayName',
                './/cda:observation/cda:code/cda:originalText/text()',
            ]

            for pattern in display_name_patterns:
                try:
                    names = entry.xpath(pattern, namespaces=ns)
                    if names and names[0].strip():
                        lab_data['display_name'] = names[0].strip()
                        break
                except:
                    continue

            # Extract code information for LLM analysis
            code_patterns = [
                './/cda:observation/cda:code',
                './/cda:code',
            ]

            for pattern in code_patterns:
                try:
                    code_elements = entry.xpath(pattern, namespaces=ns)
                    if code_elements:
                        code_elem = code_elements[0]

                        # Get code
                        code = code_elem.get('code')
                        if code and code.strip():
                            lab_data['code'] = code.strip()

                        # Get code system
                        code_system = code_elem.get('codeSystem')
                        if code_system and code_system.strip():
                            lab_data['code_system'] = code_system.strip()

                        # Get code system name
                        code_system_name = code_elem.get('codeSystemName')
                        if code_system_name and code_system_name.strip():
                            lab_data['code_system_name'] = code_system_name.strip()

                        if lab_data['code']:  # Found code info
                            break
                except:
                    continue


            value_patterns = [
                './/cda:observation/cda:value/@value',
                './/cda:value/@value',
            ]

            for pattern in value_patterns:
                try:
                    values = entry.xpath(pattern, namespaces=ns)
                    if values and values[0].strip():
                        lab_data['value'] = values[0].strip()
                        break
                except:
                    continue


            unit_patterns = [
                './/cda:observation/cda:value/@unit',
                './/cda:value/@unit',
            ]

            for pattern in unit_patterns:
                try:
                    units = entry.xpath(pattern, namespaces=ns)
                    if units and units[0].strip():
                        lab_data['unit'] = units[0].strip()
                        break
                except:
                    continue

            return lab_data if lab_data['display_name'] and lab_data['value'] else None

        except Exception as e:
            print(f"Error extracting single lab result: {e}")
            return None

    def _extract_medications(self, doc: etree._Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract medications with basic parsing."""
        structured_data = []

        med_xpath_patterns = [
            '//cda:section[cda:templateId/@root="2.16.840.1.113883.10.20.22.2.1.1"]//cda:entry',
            '//cda:section[contains(translate(cda:title/text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "medication")]//cda:entry',
            '//cda:substanceAdministration',
        ]

        med_entries = []
        for pattern in med_xpath_patterns:
            try:
                entries = doc.xpath(pattern, namespaces=ns)
                med_entries.extend(entries)
            except:
                continue

        unique_entries = self._remove_duplicate_entries(med_entries)

        for i, entry in enumerate(unique_entries):
            try:
                med_data = self._extract_single_medication(entry, ns, i)
                if med_data and med_data.get('display_name'):
                    structured_data.append(med_data)
            except Exception as e:
                print(f"⚠️ Error extracting medication entry {i}: {e}")

        return structured_data

    def _extract_single_medication(self, entry: etree._Element, ns: Dict[str, str], entry_index: int) -> Dict[str, Any]:
        """Extract a single medication with code information for LLM analysis."""
        med_data = {
            'display_name': None,
            'dose': None,
            'route': None,
            'frequency': None,
            'code': None,  # ← For LLM analysis
            'code_system': None,  # ← For LLM analysis
            'code_system_name': None,  # ← For LLM analysis
            'type': 'medication',
            'entry_index': entry_index
        }

        try:

            name_patterns = [
                './/cda:manufacturedMaterial/cda:code/@displayName',
                './/cda:manufacturedMaterial/cda:name/text()',
                './/cda:code/@displayName',
                './/cda:originalText/text()'
            ]

            for pattern in name_patterns:
                try:
                    names = entry.xpath(pattern, namespaces=ns)
                    if names and names[0].strip():
                        med_data['display_name'] = names[0].strip()
                        break
                except:
                    continue

            # Extract code information for LLM analysis
            code_patterns = [
                './/cda:manufacturedMaterial/cda:code',
                './/cda:code',
            ]

            for pattern in code_patterns:
                try:
                    code_elements = entry.xpath(pattern, namespaces=ns)
                    if code_elements:
                        code_elem = code_elements[0]

                        # Get code
                        code = code_elem.get('code')
                        if code and code.strip():
                            med_data['code'] = code.strip()

                        # Get code system
                        code_system = code_elem.get('codeSystem')
                        if code_system and code_system.strip():
                            med_data['code_system'] = code_system.strip()

                        # Get code system name
                        code_system_name = code_elem.get('codeSystemName')
                        if code_system_name and code_system_name.strip():
                            med_data['code_system_name'] = code_system_name.strip()

                        if med_data['code']:  # Found code info
                            break
                except:
                    continue

            # Extract dose information (existing logic)
            dose_patterns = [
                './/cda:doseQuantity/@value',
                './/cda:substanceAdministration/cda:doseQuantity/@value'
            ]

            for pattern in dose_patterns:
                try:
                    doses = entry.xpath(pattern, namespaces=ns)
                    if doses:
                        dose_value = doses[0].strip()
                        unit_xpath = pattern.replace('@value', '@unit')
                        units = entry.xpath(unit_xpath, namespaces=ns)
                        unit_str = f" {units[0]}" if units else ""
                        med_data['dose'] = f"{dose_value}{unit_str}"
                        break
                except:
                    continue

            return med_data if med_data['display_name'] else None

        except Exception as e:
            return None

    def _extract_problems(self, doc: etree._Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract problems/diagnoses with basic parsing."""
        structured_data = []

        problem_xpath_patterns = [
            '//cda:section[cda:templateId/@root="2.16.840.1.113883.10.20.22.2.5.1"]//cda:entry',
            '//cda:section[contains(translate(cda:title/text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "problem")]//cda:entry',
        ]

        problem_entries = []
        for pattern in problem_xpath_patterns:
            try:
                entries = doc.xpath(pattern, namespaces=ns)
                problem_entries.extend(entries)
            except:
                continue

        unique_entries = self._remove_duplicate_entries(problem_entries)

        for i, entry in enumerate(unique_entries):
            try:
                problem_data = self._extract_single_problem(entry, ns, i)
                if problem_data and problem_data.get('display_name'):
                    structured_data.append(problem_data)
            except Exception as e:
                print(f"⚠️ Error extracting problem entry {i}: {e}")

        return structured_data

    def _extract_single_problem(self, entry: etree._Element, ns: Dict[str, str], entry_index: int) -> Dict[str, Any]:
        """Extract a single problem/diagnosis with code information for LLM analysis."""
        problem_data = {
            'display_name': None,
            'status': None,
            'code': None,  # ← For LLM analysis
            'code_system': None,  # ← For LLM analysis
            'code_system_name': None,  # ← For LLM analysis
            'type': 'problem',
            'entry_index': entry_index
        }

        try:
            # Extract problem name (existing logic)
            name_patterns = [
                './/cda:value/@displayName',
                './/cda:code/@displayName',
                './/cda:originalText/text()',
            ]

            for pattern in name_patterns:
                try:
                    names = entry.xpath(pattern, namespaces=ns)
                    if names and names[0].strip():
                        problem_data['display_name'] = names[0].strip()
                        break
                except:
                    continue

            # Extract code information for LLM analysis
            code_patterns = [
                './/cda:value',
                './/cda:code',
            ]

            for pattern in code_patterns:
                try:
                    code_elements = entry.xpath(pattern, namespaces=ns)
                    if code_elements:
                        code_elem = code_elements[0]

                        # Get code
                        code = code_elem.get('code')
                        if code and code.strip():
                            problem_data['code'] = code.strip()

                        # Get code system
                        code_system = code_elem.get('codeSystem')
                        if code_system and code_system.strip():
                            problem_data['code_system'] = code_system.strip()

                        # Get code system name
                        code_system_name = code_elem.get('codeSystemName')
                        if code_system_name and code_system_name.strip():
                            problem_data['code_system_name'] = code_system_name.strip()

                        if problem_data['code']:  # Found code info
                            break
                except:
                    continue

            # Extract status (existing logic)
            status_patterns = [
                './/cda:statusCode/@code',
                './/cda:observation/cda:statusCode/@code'
            ]

            for pattern in status_patterns:
                try:
                    statuses = entry.xpath(pattern, namespaces=ns)
                    if statuses:
                        status_code = statuses[0].strip()
                        # Map status codes to readable names
                        status_map = {
                            'completed': 'Active',
                            'active': 'Active',
                            'resolved': 'Resolved',
                            'inactive': 'Inactive'
                        }
                        problem_data['status'] = status_map.get(status_code.lower(), status_code)
                        break
                except:
                    continue

            return problem_data if problem_data['display_name'] else None

        except Exception as e:
            return None

    def _remove_duplicate_entries(self, entries: List[etree._Element]) -> List[etree._Element]:
        """Remove duplicate XML entries."""
        unique_entries = []
        seen_entries = set()

        for entry in entries:
            try:
                entry_content = etree.tostring(entry, encoding='unicode')
                entry_hash = hash(entry_content)
                if entry_hash not in seen_entries:
                    unique_entries.append(entry)
                    seen_entries.add(entry_hash)
            except:
                unique_entries.append(entry)

        return unique_entries

    def extract_narrative_text(self, doc: etree._Element, section_type: str) -> str:
        """Extract narrative text from CCDA section."""
        ns = self.config["namespaces"]

        try:
            if section_type == 'lab_results':
                text_patterns = [
                    '//cda:section[cda:templateId/@root="2.16.840.1.113883.10.20.22.2.3.1"]//cda:text//text()',
                    '//cda:section[contains(translate(cda:title/text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "result")]//cda:text//text()'
                ]
            elif section_type == 'medications':
                text_patterns = [
                    '//cda:section[cda:templateId/@root="2.16.840.1.113883.10.20.22.2.1.1"]//cda:text//text()',
                    '//cda:section[contains(translate(cda:title/text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "medication")]//cda:text//text()'
                ]
            elif section_type == 'problems':
                text_patterns = [
                    '//cda:section[cda:templateId/@root="2.16.840.1.113883.10.20.22.2.5.1"]//cda:text//text()',
                    '//cda:section[contains(translate(cda:title/text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "problem")]//cda:text//text()'
                ]
            else:
                text_patterns = []

            for pattern in text_patterns:
                try:
                    text_nodes = doc.xpath(pattern, namespaces=ns)
                    if text_nodes:
                        narrative_text = ' '.join([text.strip() for text in text_nodes if text.strip()])
                        if narrative_text:
                            return narrative_text
                except:
                    continue

        except Exception as e:
            print(f"Error extracting narrative text for {section_type}: {e}")

        return ""

    def extract_patient_demographics(self, doc: etree._Element) -> Dict[str, Any]:
        """Extract patient demographics with improved parsing."""
        ns = self.config["namespaces"]
        demographics = {
            'patient_name': None,
            'age': None,
            'gender': None,
            'birth_date': None,
            'formatted_birth_date': None
        }

        try:
            print("🔍 Extracting patient demographics...")

            # Extract patient name with multiple strategies
            name_patterns = [
                '//cda:recordTarget/cda:patientRole/cda:patient/cda:name',
                '//cda:patient/cda:name',
                '//cda:ClinicalDocument/cda:recordTarget/cda:patientRole/cda:patient/cda:name'
            ]

            for pattern in name_patterns:
                try:
                    name_nodes = doc.xpath(pattern, namespaces=ns)
                    if name_nodes:
                        name_node = name_nodes[0]

                        # Extract name components
                        given_names = name_node.xpath('.//cda:given/text()', namespaces=ns)
                        family_names = name_node.xpath('.//cda:family/text()', namespaces=ns)

                        name_parts = []

                        # Add given names (first, middle, etc.)
                        if given_names:
                            name_parts.extend([name.strip() for name in given_names if name.strip()])

                        # Add family name
                        if family_names:
                            name_parts.extend([name.strip() for name in family_names if name.strip()])

                        if name_parts:
                            demographics['patient_name'] = ' '.join(name_parts)
                            print(f"✅ Patient name extracted: {demographics['patient_name']}")
                            break
                        else:
                            # Try alternative extraction - sometimes names are in text nodes
                            name_text = name_node.xpath('.//text()', namespaces=ns)
                            if name_text:
                                clean_text = [text.strip() for text in name_text if text.strip()]
                                if clean_text:
                                    demographics['patient_name'] = ' '.join(clean_text)
                                    print(f"✅ Patient name extracted (alternative): {demographics['patient_name']}")
                                    break
                except Exception as e:
                    print(f"⚠️ Error with name pattern {pattern}: {e}")
                    continue

            # Extract birth date with enhanced parsing
            birth_date_patterns = [
                '//cda:recordTarget/cda:patientRole/cda:patient/cda:birthTime/@value',
                '//cda:patient/cda:birthTime/@value',
                '//cda:ClinicalDocument/cda:recordTarget/cda:patientRole/cda:patient/cda:birthTime/@value'
            ]

            birth_date_str = None
            for pattern in birth_date_patterns:
                try:
                    birth_dates = doc.xpath(pattern, namespaces=ns)
                    if birth_dates and birth_dates[0].strip():
                        birth_date_str = birth_dates[0].strip()
                        print(f"✅ Birth date string extracted: {birth_date_str}")
                        break
                except Exception as e:
                    print(f"⚠️ Error with birth date pattern {pattern}: {e}")
                    continue

            # Enhanced birth date parsing and age calculation
            if birth_date_str:
                try:
                    # Handle different birth date formats
                    birth_date_clean = birth_date_str.replace('-', '').replace('/', '').replace(' ', '')

                    # Extract date components
                    if len(birth_date_clean) >= 8:
                        year_str = birth_date_clean[:4]
                        month_str = birth_date_clean[4:6] if len(birth_date_clean) >= 6 else "01"
                        day_str = birth_date_clean[6:8] if len(birth_date_clean) >= 8 else "01"

                        # Validate and convert
                        birth_year = int(year_str)
                        birth_month = max(1, min(12, int(month_str)))  # Ensure valid month
                        birth_day = max(1, min(31, int(day_str)))  # Ensure valid day

                        # Validate year range
                        current_year = datetime.now().year
                        if 1900 <= birth_year <= current_year:

                            # Create birth date
                            try:
                                birth_date = datetime(birth_year, birth_month, birth_day)
                                demographics['birth_date'] = birth_date.strftime("%Y-%m-%d")
                                demographics['formatted_birth_date'] = birth_date.strftime("%B %d, %Y")

                                # Calculate age more accurately
                                today = datetime.now()
                                age = today.year - birth_date.year

                                # Adjust if birthday hasn't occurred this year
                                if (today.month, today.day) < (birth_date.month, birth_date.day):
                                    age -= 1

                                # Validate age range
                                if 0 <= age <= 150:
                                    demographics['age'] = age
                                    print(
                                        f"✅ Age calculated: {age} years (born {demographics['formatted_birth_date']})")
                                else:
                                    print(f"⚠️ Invalid age calculated: {age} years")

                            except ValueError as e:
                                print(f"⚠️ Invalid date components: {birth_year}-{birth_month}-{birth_day}, Error: {e}")

                        else:
                            print(f"⚠️ Invalid birth year: {birth_year}")

                    else:
                        print(f"⚠️ Birth date string too short: {birth_date_str}")

                except Exception as e:
                    print(f"⚠️ Error parsing birth date '{birth_date_str}': {e}")

            # Extract gender with multiple strategies
            gender_patterns = [
                '//cda:recordTarget/cda:patientRole/cda:patient/cda:administrativeGenderCode/@code',
                '//cda:patient/cda:administrativeGenderCode/@code',
                '//cda:ClinicalDocument/cda:recordTarget/cda:patientRole/cda:patient/cda:administrativeGenderCode/@code',
                '//cda:recordTarget/cda:patientRole/cda:patient/cda:administrativeGenderCode/@displayName',
                '//cda:patient/cda:administrativeGenderCode/@displayName'
            ]

            for pattern in gender_patterns:
                try:
                    genders = doc.xpath(pattern, namespaces=ns)
                    if genders and genders[0].strip():
                        gender_value = genders[0].strip().upper()

                        # Map gender codes and names
                        gender_mapping = {
                            'M': 'Male',
                            'F': 'Female',
                            'MALE': 'Male',
                            'FEMALE': 'Female',
                            'UN': 'Unknown',
                            'UNK': 'Unknown'
                        }

                        demographics['gender'] = gender_mapping.get(gender_value, gender_value)
                        print(f"✅ Gender extracted: {demographics['gender']}")
                        break
                except Exception as e:
                    print(f"⚠️ Error with gender pattern {pattern}: {e}")
                    continue

            # Summary
            print(f"📋 Final demographics:")
            print(f"   Name: {demographics.get('patient_name', 'Not found')}")
            print(f"   Age: {demographics.get('age', 'Not calculated')} years")
            print(f"   Gender: {demographics.get('gender', 'Not found')}")
            print(f"   Birth Date: {demographics.get('formatted_birth_date', 'Not found')}")

        except Exception as e:
            print(f"❌ Error extracting patient demographics: {e}")
            import traceback
            traceback.print_exc()

        return demographics

    def _extract_document_metadata(self, doc: etree._Element) -> Dict[str, Any]:
        """Extract document metadata."""
        ns = self.config["namespaces"]
        metadata = {
            'title': None,
            'effective_time': None,
            'author': None,
            'custodian': None,
            'document_type': None
        }

        try:
            # Extract document title
            title_patterns = [
                '//cda:ClinicalDocument/cda:title/text()',
                '//cda:title/text()'
            ]

            for pattern in title_patterns:
                try:
                    titles = doc.xpath(pattern, namespaces=ns)
                    if titles and titles[0].strip():
                        metadata['title'] = titles[0].strip()
                        break
                except:
                    continue

            # Extract effective time
            time_patterns = [
                '//cda:ClinicalDocument/cda:effectiveTime/@value',
                '//cda:effectiveTime/@value'
            ]

            for pattern in time_patterns:
                try:
                    times = doc.xpath(pattern, namespaces=ns)
                    if times and times[0].strip():
                        time_str = times[0].strip()
                        if len(time_str) >= 8:
                            formatted_date = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]}"
                            metadata['effective_time'] = formatted_date
                        break
                except:
                    continue

        except Exception as e:
            print(f"Error extracting document metadata: {e}")

        return metadata

    def _extract_document_id(self, doc: etree._Element) -> str:
        """Extract document ID from CCDA."""
        try:
            ns = self.config["namespaces"]
            id_patterns = [
                '//cda:ClinicalDocument/cda:id/@extension',
                '//cda:id/@extension'
            ]

            for pattern in id_patterns:
                try:
                    id_nodes = doc.xpath(pattern, namespaces=ns)
                    if id_nodes and id_nodes[0].strip():
                        return id_nodes[0].strip()
                except:
                    continue

        except Exception as e:
            print(f"Error extracting document ID: {e}")

        return "unknown_document"

    def _extract_patient_id(self, doc: etree._Element) -> str:
        """Extract patient ID from CCDA."""
        try:
            ns = self.config["namespaces"]
            patient_id_patterns = [
                '//cda:ClinicalDocument/cda:recordTarget/cda:patientRole/cda:id/@extension',
                '//cda:recordTarget/cda:patientRole/cda:id/@extension'
            ]

            for pattern in patient_id_patterns:
                try:
                    id_nodes = doc.xpath(pattern, namespaces=ns)
                    if id_nodes and id_nodes[0].strip():
                        return id_nodes[0].strip()
                except:
                    continue

        except Exception as e:
            print(f"Error extracting patient ID: {e}")

        return "unknown_patient"


    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _create_default_section_score(self, section_type: str, section_present: bool) -> Dict[str, Any]:
        """Create default section score when section is missing or empty."""
        if not section_present:
            return {
                'section_present': False,
                'overall_score': 0.0,
                'quality_level': 'Missing',
                'category_scores': {
                    'completeness': 0.0,
                    'structural_integrity': 0.0,
                    'clinical_plausibility': 0.0,
                    'narrative_consistency': 0.0
                },
                'issues': [f"{section_type.replace('_', ' ').title()} section not found in document"],
                'recommendations': [f"Add {section_type.replace('_', ' ').title()} section to document"],
                'clinical_details': {'ai_model_used': 'None', 'analysis_type': 'Section Missing'}
            }
        else:
            return {
                'section_present': True,
                'overall_score': 0.0,
                'quality_level': 'Poor',
                'category_scores': {
                    'completeness': 0.0,
                    'structural_integrity': 0.0,
                    'clinical_plausibility': 0.0,
                    'narrative_consistency': 0.0
                },
                'issues': [f"{section_type.replace('_', ' ').title()} section is empty"],
                'recommendations': [f"Add content to {section_type.replace('_', ' ').title()} section"],
                'clinical_details': {'ai_model_used': 'None', 'analysis_type': 'Section Empty'}
            }

    def _create_error_section_score(self, section_type: str, error_msg: str) -> Dict[str, Any]:
        """Create error section score when processing fails."""
        return {
            'section_present': False,
            'overall_score': 0.0,
            'quality_level': 'Error',
            'category_scores': {
                'completeness': 0.0,
                'structural_integrity': 0.0,
                'clinical_plausibility': 0.0,
                'narrative_consistency': 0.0
            },
            'issues': [f"Error processing {section_type.replace('_', ' ').title()} section: {error_msg}"],
            'recommendations': [f"Review {section_type.replace('_', ' ').title()} section structure and content"],
            'clinical_details': {'ai_model_used': 'None', 'analysis_type': 'Processing Error', 'error': error_msg}
        }

    def _compile_section_score(self, scores: Dict[str, float], issues: List[str],
                               recommendations: List[str], section_type: str,
                               section_present: bool) -> Dict[str, Any]:
        """Compile final section score."""
        # Calculate overall score
        score_values = [max(0, min(100, score)) for score in scores.values()]
        overall_score = sum(score_values) / len(score_values) if score_values else 0

        # Determine quality level
        if overall_score >= 85:
            quality_level = "High"
        elif overall_score >= 70:
            quality_level = "Medium"
        elif overall_score >= 50:
            quality_level = "Low"
        else:
            quality_level = "Poor"

        return {
            'section_present': section_present,
            'overall_score': overall_score,
            'quality_level': quality_level,
            'category_scores': scores,
            'issues': issues,
            'recommendations': recommendations
        }

    # ========================================================================
    # REPORT GENERATION
    # ========================================================================

    def generate_report(self, score_report: Dict[str, Any]) -> str:
        """Generate a comprehensive text report from Pure LLM score results."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LLM-ENHANCED CCDA DOCUMENT QUALITY REPORT")
        report_lines.append("=" * 80)

        # Document info
        report_lines.append(f"Document ID: {score_report.get('document_id', 'Unknown')}")
        report_lines.append(f"Patient ID: {score_report.get('patient_id', 'Unknown')}")
        report_lines.append(f"Analysis Timestamp: {score_report.get('timestamp', 'Unknown')}")

        # Analysis configuration
        config = score_report.get('analysis_config', {})
        report_lines.append(f"AI Analysis: {'✅ Enabled' if config.get('ai_enabled') else '❌ Disabled'}")
        report_lines.append(f"Analyzer Type: {config.get('analyzer_type', 'Unknown')}")
        report_lines.append("")

        # Overall score
        overall_score = score_report.get('overall_score', 0)
        quality_level = score_report.get('quality_level', 'Unknown')
        report_lines.append(f"🎯 OVERALL QUALITY SCORE: {overall_score:.1f}/100 ({quality_level})")

        # Quality indicator
        if overall_score >= 85:
            report_lines.append("🟢 Excellent clinical documentation quality")
        elif overall_score >= 70:
            report_lines.append("🟡 Good clinical documentation quality")
        elif overall_score >= 50:
            report_lines.append("🟠 Moderate clinical documentation quality")
        else:
            report_lines.append("🔴 Poor clinical documentation quality - needs improvement")

        report_lines.append("")

        # Patient demographics
        demographics = score_report.get('patient_demographics', {})
        if demographics:
            report_lines.append("👤 PATIENT INFORMATION:")
            if demographics.get('patient_name'):
                report_lines.append(f"  Name: {demographics['patient_name']}")
            if demographics.get('age'):
                report_lines.append(f"  Age: {demographics['age']} years")
            if demographics.get('gender'):
                report_lines.append(f"  Gender: {demographics['gender']}")
            report_lines.append("")

        # Section details
        sections = score_report.get('sections', {})
        for section_name, section_data in sections.items():
            report_lines.append("-" * 60)
            report_lines.append(f"{section_name.replace('_', ' ').title()} Section")
            report_lines.append("-" * 60)

            if section_data.get('section_present', False):
                section_score = section_data.get('overall_score', 0)
                section_quality = section_data.get('quality_level', 'Unknown')
                report_lines.append(f"Score: {section_score:.1f}/100 ({section_quality})")

                # Category scores
                category_scores = section_data.get('category_scores', {})
                report_lines.append("\n📊 LLM Category Scores:")
                for category, score in category_scores.items():
                    emoji = "🤖"
                    report_lines.append(f"  {emoji} {category.replace('_', ' ').title()}: {score:.1f}/100")

                # Issues
                issues = section_data.get('issues', [])
                if issues:
                    report_lines.append("\n⚠️ Issues Identified:")
                    for issue in issues[:10]:  # Limit to first 10 issues
                        report_lines.append(f"  - {issue}")
                    if len(issues) > 10:
                        report_lines.append(f"  ... and {len(issues) - 10} more issues")

                # Recommendations
                recommendations = section_data.get('recommendations', [])
                if recommendations:
                    report_lines.append("\n💡 Recommendations:")
                    for rec in recommendations[:10]:  # Limit to first 10 recommendations
                        report_lines.append(f"  • {rec}")
                    if len(recommendations) > 10:
                        report_lines.append(f"  ... and {len(recommendations) - 10} more recommendations")
            else:
                report_lines.append("Status: ❌ Section not present")

            report_lines.append("")

        # Summary
        report_lines.append("=" * 80)
        report_lines.append("📊 ANALYSIS SUMMARY")
        report_lines.append("=" * 80)

        # Count sections present
        sections_present = sum(1 for s in sections.values() if s.get('section_present', False))
        total_sections = len(sections)
        report_lines.append(f"Sections Present: {sections_present}/{total_sections}")

        # Pure LLM analysis summary
        llm_sections = sum(1 for s in sections.values() if
                           'Pure LLM' in s.get('clinical_details', {}).get('ai_model_used', ''))
        if llm_sections > 0:
            report_lines.append(f"🤖 LLM Analysis: {llm_sections} sections analyzed")
            report_lines.append("🎯 NO Knowledge Base: ALL analysis via AI reasoning")
            report_lines.append("🧠 Drug Interactions: AI-detected without databases")

        # Performance indicators
        if overall_score >= 85:
            report_lines.append("\n✅ EXCELLENT: Document meets high-quality clinical standards")
        elif overall_score >= 70:
            report_lines.append("\n👍 GOOD: Document meets basic clinical standards with room for improvement")
        elif overall_score >= 50:
            report_lines.append("\n⚠️ MODERATE: Document needs significant improvements to meet clinical standards")
        else:
            report_lines.append("\n❌ POOR: Document requires major revisions to meet clinical documentation standards")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("Report generated by Pure LLM-Enhanced CCDA Section Quality Scorer")
        report_lines.append("Powered by OpenAI GPT with ZERO knowledge base dependency")
        report_lines.append("ALL clinical reasoning via AI - NO hardcoded medical rules")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def generate_json_report(self, score_report: Dict[str, Any]) -> str:
        """Generate a JSON report from Pure LLM score results."""
        return json.dumps(score_report, indent=2, default=str)


"""
Missing Functions for Complete CCDA Analysis System
===================================================

Add these functions to complete your dataquality_analyzer.py file.
Your existing main() function is fine - these are just the missing utility functions.
"""


def print_analysis_summary(score_report: Dict[str, Any], processing_time: float, verbose: bool = False):
    """Print analysis summary to console."""
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)

    # Document info
    doc_id = score_report.get('document_id', 'Unknown')
    patient_id = score_report.get('patient_id', 'Unknown')
    overall_score = score_report.get('overall_score', 0)
    quality_level = score_report.get('quality_level', 'Unknown')

    print(f"📄 Document ID: {doc_id}")
    print(f"👤 Patient ID: {patient_id}")
    print(f"⏱️  Processing Time: {processing_time:.1f} seconds")
    print(f"🎯 Overall Score: {overall_score:.1f}/100 ({quality_level})")

    # Quality indicator
    if overall_score >= 80:
        print("🟢 Excellent clinical documentation quality")
    elif overall_score >= 60:
        print("🟡 Good clinical documentation quality")
    else:
        print("🔴 Poor clinical documentation quality - needs improvement")

    # AI Analysis info
    config = score_report.get('analysis_config', {})
    if config.get('ai_enabled'):
        print("🤖 Analysis Type: Pure LLM Enhanced (NO Knowledge Base)")
    else:
        print("📋 Analysis Type: Rule-Based Only")

    # Section summary
    sections = score_report.get('sections', {})
    print(f"\n📊 SECTION SCORES:")

    for section_name, section_data in sections.items():
        if section_data.get('section_present', False):
            section_score = section_data.get('overall_score', 0)
            quality = section_data.get('quality_level', 'Unknown')
            print(f"  {section_name.replace('_', ' ').title()}: {section_score:.1f}/100 ({quality})")

            if verbose:
                # Show dimension scores
                category_scores = section_data.get('category_scores', {})
                for dimension, score in category_scores.items():
                    print(f"     {dimension.replace('_', ' ').title()}: {score:.1f}")
        else:
            print(f"  {section_name.replace('_', ' ').title()}: ❌ Not Present")

    # Critical findings summary
    if verbose:
        print(f"\n🔍 DETAILED FINDINGS:")
        for section_name, section_data in sections.items():
            if section_data.get('section_present', False):
                issues = section_data.get('issues', [])
                if issues:
                    print(f"\n{section_name.replace('_', ' ').title()} Issues:")
                    for issue in issues[:3]:  # Show first 3 issues
                        print(f"  • {issue}")
                    if len(issues) > 3:
                        print(f"  ... and {len(issues) - 3} more issues")


def export_single_reports(score_report: Dict[str, Any], scorer, file_path, args):
    """Export reports for single document analysis."""
    from pathlib import Path

    output_dir = Path(args.output) if args.output else Path(file_path).parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(file_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exported_files = []

    try:
        # Export JSON report
        if args.json or not hasattr(args, 'json'):
            json_file = output_dir / f"{base_name}_report_{timestamp}.json"
            json_report = scorer.generate_json_report(score_report)

            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(json_report)

            exported_files.append(str(json_file))
            print(f"📄 JSON report saved: {json_file}")

        # Export text report
        if not args.json or not hasattr(args, 'json'):
            text_file = output_dir / f"{base_name}_report_{timestamp}.txt"
            text_report = scorer.generate_report(score_report)

            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_report)

            exported_files.append(str(text_file))
            print(f"📝 Text report saved: {text_file}")

        print(f"✅ Reports exported to: {output_dir}")

    except Exception as e:
        print(f"⚠️ Error exporting reports: {e}")


def display_system_info():
    """Display system information and capabilities."""
    print("🏥 CCDA Data Quality Analyzer - System Information")
    print("=" * 60)

    # Check AI capabilities
    try:
        import os
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')

        print(f"🤖 OpenAI API: {'✅ Available' if openai_key else '❌ No API Key'}")
        print(f"🤖 Anthropic API: {'✅ Available' if anthropic_key else '❌ No API Key'}")

        if openai_key or anthropic_key:
            print("🧠 LLM Analysis: ✅ Enabled")
            print("📚 Knowledge Base: ❌ None (Pure LLM Mode)")
        else:
            print("🧠 LLM Analysis: ⚠️ No API Keys")
            print("📋 Fallback Mode: Rule-based analysis only")

    except Exception as e:
        print(f"⚠️ Error checking AI capabilities: {e}")

    # System capabilities
    print(f"\n🔍 ANALYSIS CAPABILITIES:")
    print(f"📊 Sections Supported: Lab Results, Medications, Problems")
    print(f"🎯 Quality Dimensions: Completeness, Structural Integrity, Clinical Plausibility, Narrative Consistency")
    print(f"🤖 Analysis Mode: Pure LLM (NO hardcoded knowledge base)")
    print(f"💊 Drug Interactions: AI-detected without databases")
    print(f"🔬 Lab Analysis: AI clinical reasoning without reference ranges")
    print(f"🏥 Problem Analysis: AI diagnostic assessment without rule engines")

    # Dependencies
    print(f"\n📦 DEPENDENCIES:")

    try:
        import lxml
        print(f"✅ lxml: {lxml.__version__}")
    except ImportError:
        print("❌ lxml: Not available")

    try:
        import openai
        print(f"✅ openai: Available")
    except ImportError:
        print("❌ openai: Not available")

    try:
        import streamlit
        print(f"✅ streamlit: {streamlit.__version__} (Web UI available)")
    except ImportError:
        print("❌ streamlit: Not available (Web UI disabled)")


def test_ai_connectivity():
    """Test AI service connectivity."""
    print("🤖 Testing AI Connectivity...")

    try:
        # Test OpenAI
        openai_analyzer = GenerativeClinicalAnalyzer(provider="openai")
        if openai_analyzer.enabled:
            print("✅ OpenAI: Connection available")
        else:
            print("❌ OpenAI: No API key or connection failed")

        # Test Claude (placeholder)
        claude_analyzer = GenerativeClinicalAnalyzer(provider="claude")
        if claude_analyzer.enabled:
            print("✅ Claude: Connection available")
        else:
            print("❌ Claude: No API key or connection failed")

    except Exception as e:
        print(f"❌ AI connectivity test failed: {e}")
        return False

    return True


def process_batch_documents(input_dir, output_dir, pattern="*.xml", no_ai=False, continue_on_error=False):
    """Process multiple CCDA documents in batch."""
    from pathlib import Path
    import glob

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return False

    # Find files to process
    files = list(input_path.glob(pattern))
    if not files:
        print(f"❌ No files found matching pattern '{pattern}' in {input_path}")
        return False

    print(f"📁 Found {len(files)} files to process")
    print(f"📤 Output directory: {output_path}")
    print(f"🤖 AI Analysis: {'Disabled' if no_ai else 'Enabled'}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup configuration
    config = {}
    if no_ai:
        config['use_ai_analysis'] = False

    # Process files
    results = []
    failed_files = []
    total_processing_time = 0

    for i, file_path in enumerate(files, 1):
        print(f"\n📄 Processing {i}/{len(files)}: {file_path.name}")

        try:
            start_time = time.time()
            scorer = CCDASectionScorer(config)
            score_report = scorer.score_ccda(str(file_path))
            processing_time = time.time() - start_time
            total_processing_time += processing_time

            if 'error' in score_report:
                print(f"❌ Failed: {score_report['error']}")
                failed_files.append({
                    'file': str(file_path),
                    'error': score_report['error']
                })

                if not continue_on_error:
                    print("🛑 Stopping due to error")
                    break
                continue

            # Store result
            result = {
                'file': str(file_path),
                'overall_score': score_report.get('overall_score', 0),
                'quality_level': score_report.get('quality_level', 'Unknown'),
                'processing_time': processing_time,
                'sections': {}
            }

            # Extract section scores
            sections = score_report.get('sections', {})
            for section_name, section_data in sections.items():
                if section_data.get('section_present', False):
                    result['sections'][section_name] = section_data.get('overall_score', 0)

            results.append(result)

            print(f"✅ Score: {score_report.get('overall_score', 0):.1f}/100 ({processing_time:.1f}s)")

            # Save individual reports
            base_name = file_path.stem

            # JSON report
            json_file = output_path / f"{base_name}_report.json"
            json_report = scorer.generate_json_report(score_report)
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(json_report)

            # Text report
            text_file = output_path / f"{base_name}_report.txt"
            text_report = scorer.generate_report(score_report)
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_report)

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            failed_files.append({
                'file': str(file_path),
                'error': str(e)
            })

            if not continue_on_error:
                print("🛑 Stopping due to error")
                break

    # Generate batch summary
    generate_batch_summary(results, failed_files, output_path, total_processing_time)

    success_count = len(results)
    total_count = len(files)

    print(f"\n📊 BATCH COMPLETE: {success_count}/{total_count} files processed successfully")
    print(f"⏱️  Total processing time: {total_processing_time:.1f} seconds")

    return len(failed_files) == 0


def generate_batch_summary(results, failed_files, output_dir, total_time):
    """Generate batch processing summary."""
    from pathlib import Path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(output_dir) / f"batch_summary_{timestamp}.json"

    # Calculate statistics
    if results:
        scores = [r['overall_score'] for r in results]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        # Quality distribution
        high_quality = sum(1 for s in scores if s >= 80)
        medium_quality = sum(1 for s in scores if 60 <= s < 80)
        low_quality = sum(1 for s in scores if s < 60)
    else:
        avg_score = min_score = max_score = 0
        high_quality = medium_quality = low_quality = 0

    summary = {
        'batch_info': {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(results) + len(failed_files),
            'successful_analyses': len(results),
            'failed_analyses': len(failed_files),
            'total_processing_time_seconds': total_time
        },
        'quality_statistics': {
            'average_score': round(avg_score, 1),
            'min_score': round(min_score, 1),
            'max_score': round(max_score, 1),
            'high_quality_count': high_quality,
            'medium_quality_count': medium_quality,
            'low_quality_count': low_quality
        },
        'detailed_results': results,
        'failed_files': failed_files
    }

    # Save summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"📊 Batch summary saved: {summary_file}")

