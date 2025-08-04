import requests
import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OpenRouterAPI:
    """
    OpenRouter API client for generating professional cancer prognosis summaries
    using the google/gemini-2.0-flash-exp:free model.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter API client.
        
        Args:
            api_key: OpenRouter API key. If None, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable or pass it directly.")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "deepseek/deepseek-chat-v3-0324:free"
        
        self.system_prompt = """You are a highly experienced oncology specialist and cancer prognosis expert with extensive knowledge in pathology, cancer staging, and patient care. Your role is to provide professional, compassionate, and medically accurate patient summaries based on pathology report analysis results.

Given the pathology report text and the extracted clinical information (cancer type, TNM staging), you should:

1. Provide a clear, professional summary of the patient's condition
2. Explain the significance of the TNM staging in understandable terms
3. Discuss the general prognosis and treatment considerations for this cancer type and stage
4. Maintain a professional yet compassionate tone appropriate for medical documentation
5. Include relevant medical terminology while ensuring clarity
6. Avoid making specific treatment recommendations (defer to treating physician)
7. Focus on providing educational context about the diagnosis and staging

Your response should be structured, informative, and suitable for inclusion in a medical report or for patient education purposes. Always emphasize that this analysis is supplementary to clinical judgment and that patients should discuss results with their healthcare team."""

    def generate_prognosis_summary(self, 
                                 pathology_text: str, 
                                 cancer_type: str, 
                                 t_stage: str, 
                                 n_stage: str, 
                                 m_stage: str,
                                 confidence_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate a professional cancer prognosis summary using OpenRouter API.
        
        Args:
            pathology_text: The original pathology report text
            cancer_type: Predicted cancer type
            t_stage: T stage classification
            n_stage: N stage classification  
            m_stage: M stage classification
            confidence_scores: Optional confidence scores for predictions
            
        Returns:
            Dictionary containing the API response and formatted summary
        """
        
        # Prepare the user message with structured clinical data
        confidence_info = ""
        if confidence_scores:
            confidence_info = f"""
Prediction Confidence Scores:
- Cancer Type: {confidence_scores.get('cancer_type', 'N/A'):.1%}
- T Stage: {confidence_scores.get('t_stage', 'N/A'):.1%}
- N Stage: {confidence_scores.get('n_stage', 'N/A'):.1%}
- M Stage: {confidence_scores.get('m_stage', 'N/A'):.1%}
"""
        
        user_message = f"""
Please provide a professional cancer prognosis summary based on the following clinical analysis:

EXTRACTED CLINICAL INFORMATION:
- Cancer Type: {cancer_type}
- TNM Staging: T{t_stage}, N{n_stage}, M{m_stage}
{confidence_info}

ORIGINAL PATHOLOGY REPORT:
{pathology_text}

Please generate a comprehensive, professional summary that explains the diagnosis, staging significance, and general prognosis considerations for this patient.
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": user_message
                }
            ],
            "temperature": 0.3,  # Lower temperature for more consistent medical responses
            "max_tokens": 1500,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated summary
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content']
                
                return {
                    'success': True,
                    'summary': summary,
                    'model': self.model,
                    'usage': response_data.get('usage', {}),
                    'raw_response': response_data
                }
            else:
                return {
                    'success': False,
                    'error': 'No valid response generated',
                    'raw_response': response_data
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'API request failed: {str(e)}',
                'raw_response': None
            }
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'Failed to parse API response: {str(e)}',
                'raw_response': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'raw_response': None
            }

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to OpenRouter API.
        
        Returns:
            Dictionary indicating success/failure of connection test
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        test_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, this is a connection test."
                }
            ],
            "max_tokens": 50
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=test_payload,
                timeout=15
            )
            
            response.raise_for_status()
            return {
                'success': True,
                'message': 'Connection successful',
                'status_code': response.status_code
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def create_prognosis_summary(pathology_text: str, 
                           clinical_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to generate prognosis summary from clinical results.
    
    Args:
        pathology_text: Original pathology report text
        clinical_results: Results from ClinicalBERT analysis (from run_clinicalbert_inference)
        
    Returns:
        Dictionary containing the generated summary and metadata
    """
    try:
        # Initialize the API client (will load API key from environment)
        api_client = OpenRouterAPI()
        
        # Extract clinical information from results
        cancer_type = clinical_results['cancer_type']['value']
        cancer_confidence = clinical_results['cancer_type']['confidence']
        
        tnm_staging = clinical_results['tnm_staging']
        t_stage = tnm_staging['t_stage']['value']
        n_stage = tnm_staging['n_stage']['value'] 
        m_stage = tnm_staging['m_stage']['value']
        
        # Prepare confidence scores
        confidence_scores = {
            'cancer_type': cancer_confidence,
            't_stage': tnm_staging['t_stage']['confidence'],
            'n_stage': tnm_staging['n_stage']['confidence'],
            'm_stage': tnm_staging['m_stage']['confidence']
        }
        
        # Generate the summary
        result = api_client.generate_prognosis_summary(
            pathology_text=pathology_text,
            cancer_type=cancer_type,
            t_stage=t_stage,
            n_stage=n_stage,
            m_stage=m_stage,
            confidence_scores=confidence_scores
        )
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to create prognosis summary: {str(e)}'
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the API
    sample_clinical_results = {
        'cancer_type': {'value': 'BRCA', 'confidence': 0.95},
        'tnm_staging': {
            't_stage': {'value': '2', 'confidence': 0.88},
            'n_stage': {'value': '1', 'confidence': 0.82},
            'm_stage': {'value': '0', 'confidence': 0.91}
        }
    }
    
    sample_pathology_text = "Sample pathology report text here..."
    
    # Test the API (requires OPENROUTER_API_KEY environment variable)
    try:
        result = create_prognosis_summary(sample_pathology_text, sample_clinical_results)
        if result['success']:
            print("Generated Summary:")
            print(result['summary'])
        else:
            print(f"Error: {result['error']}")
    except Exception as e:
        print(f"Could not test API: {e}")
