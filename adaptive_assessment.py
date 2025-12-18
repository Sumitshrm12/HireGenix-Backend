"""
Adaptive Assessment Engine
Generates real-world coding/scenario challenges that adapt to the candidate's skill level in real-time.
Focuses on practical application rather than theoretical trivia.
Uses DSPy for robust, structured challenge generation and evaluation.
"""

import os
import json
import dspy
from typing import Dict, Any, List, Optional
from enum import Enum

# Configure DSPy with Azure OpenAI
lm = dspy.LM(
    model=f"azure/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')}",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    temperature=0.4
)
dspy.settings.configure(lm=lm)

class AssessmentType(str, Enum):
    CODING = "coding"
    SYSTEM_DESIGN = "system_design"
    DEBUGGING = "debugging"
    CODE_REVIEW = "code_review"

class ChallengeGeneratorSignature(dspy.Signature):
    """Generates a practical, real-world technical challenge."""
    
    skill = dspy.InputField(desc="The specific technical skill to assess")
    difficulty = dspy.InputField(desc="Difficulty level (Junior/Mid/Senior/Expert)")
    market_context = dspy.InputField(desc="Relevant market trends to incorporate")
    
    title = dspy.OutputField(desc="Engaging challenge title")
    description = dspy.OutputField(desc="Detailed problem statement with business context")
    type = dspy.OutputField(desc="coding/system_design/debugging")
    starter_code = dspy.OutputField(desc="Initial code snippet or boilerplate")
    test_cases = dspy.OutputField(desc="JSON list of test cases with inputs and expected outputs")
    evaluation_criteria = dspy.OutputField(desc="List of specific criteria for success")
    hints = dspy.OutputField(desc="List of progressive hints")

class EvaluationSignature(dspy.Signature):
    """Evaluates a code submission against a challenge."""
    
    challenge_title = dspy.InputField(desc="Title of the challenge")
    challenge_description = dspy.InputField(desc="Description of the challenge")
    criteria = dspy.InputField(desc="Evaluation criteria")
    submission_code = dspy.InputField(desc="Candidate's code submission")
    language = dspy.InputField(desc="Programming language used")
    
    score = dspy.OutputField(desc="Float score 0-100")
    status = dspy.OutputField(desc="Pass/Fail")
    correctness_analysis = dspy.OutputField(desc="Detailed analysis of logic correctness")
    code_quality_analysis = dspy.OutputField(desc="Analysis of style, efficiency, and readability")
    security_issues = dspy.OutputField(desc="List of potential security vulnerabilities")
    optimization_suggestions = dspy.OutputField(desc="List of suggestions for improvement")
    feedback = dspy.OutputField(desc="Constructive feedback for the candidate")

class AdaptiveAssessmentEngine:
    """
    Generates and evaluates technical assessments dynamically using DSPy.
    """
    
    def __init__(self):
        self.generator = dspy.ChainOfThought(ChallengeGeneratorSignature)
        self.evaluator = dspy.ChainOfThought(EvaluationSignature)

    async def generate_challenge(
        self, 
        skill: str, 
        difficulty: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates a practical challenge based on skill and difficulty.
        """
        
        market_context_str = ""
        if context and context.get('market_trends'):
            market_context_str = f"Incorporate these modern trends: {', '.join(context['market_trends'][:2])}"

        try:
            prediction = self.generator(
                skill=skill,
                difficulty=difficulty,
                market_context=market_context_str
            )
            
            # Parse test cases if string
            test_cases = prediction.test_cases
            if isinstance(test_cases, str):
                try:
                    test_cases = json.loads(test_cases)
                except:
                    pass # Keep as string if parsing fails

            return {
                "title": prediction.title,
                "description": prediction.description,
                "type": prediction.type,
                "starter_code": prediction.starter_code,
                "test_cases": test_cases,
                "evaluation_criteria": prediction.evaluation_criteria.split(',') if isinstance(prediction.evaluation_criteria, str) else prediction.evaluation_criteria,
                "hints": prediction.hints.split(',') if isinstance(prediction.hints, str) else prediction.hints
            }
            
        except Exception as e:
            print(f"❌ Challenge generation failed: {e}")
            return {
                "error": "Failed to generate challenge",
                "details": str(e)
            }

    async def evaluate_submission(
        self, 
        challenge: Dict[str, Any], 
        submission: str, 
        language: str
    ) -> Dict[str, Any]:
        """
        Evaluates a candidate's solution using AI.
        """
        try:
            prediction = self.evaluator(
                challenge_title=challenge['title'],
                challenge_description=challenge['description'],
                criteria=', '.join(challenge['evaluation_criteria']),
                submission_code=submission,
                language=language
            )
            
            return {
                "score": float(prediction.score),
                "status": prediction.status,
                "correctness": prediction.correctness_analysis,
                "code_quality": prediction.code_quality_analysis,
                "feedback": prediction.feedback,
                "security_issues": prediction.security_issues.split(',') if isinstance(prediction.security_issues, str) else prediction.security_issues,
                "optimization_suggestions": prediction.optimization_suggestions.split(',') if isinstance(prediction.optimization_suggestions, str) else prediction.optimization_suggestions
            }
            
        except Exception as e:
            return {"error": "Evaluation failed", "details": str(e)}

# Singleton
_assessment_engine = None

def get_assessment_engine() -> AdaptiveAssessmentEngine:
    global _assessment_engine
    if _assessment_engine is None:
        _assessment_engine = AdaptiveAssessmentEngine()
    return _assessment_engine