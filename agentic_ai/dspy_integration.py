"""
DSPy Integration for Advanced Agentic AI in Python
Provides program synthesis and optimization for LLM-powered agents
"""

import os
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import dspy
from langchain_openai import AzureChatOpenAI
from .config import get_config


@dataclass
class DSPySignature:
    """Defines input/output specification for LLM programs"""
    name: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    description: str
    
    def to_prompt(self, input_values: Dict[str, Any]) -> str:
        """Convert signature to prompt"""
        input_prompts = "\n".join(
            f"{key} ({desc}): {input_values.get(key, 'N/A')}"
            for key, desc in self.inputs.items()
        )
        
        output_spec = "\n".join(
            f"{key}: {desc}"
            for key, desc in self.outputs.items()
        )
        
        return f"""{self.description}

Inputs:
{input_prompts}

Required Outputs:
{output_spec}

Provide the outputs in JSON format."""


class DSPyModule:
    """Composable LLM program component using DSPy"""
    
    def __init__(self, signature: DSPySignature, temperature: float = 0.7):
        self.signature = signature
        self.temperature = temperature
        self.examples: List[Dict[str, Any]] = []
        
        # Initialize Azure OpenAI LLM
        config = get_config()
        self.llm = AzureChatOpenAI(
            openai_api_key=config.azure.api_key,
            azure_endpoint=config.azure.endpoint,
            deployment_name=config.azure.deployment_name,
            openai_api_version=config.azure.api_version,
            temperature=temperature,
            max_tokens=4000
        )
    
    def add_example(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """Add few-shot example"""
        self.examples.append({"inputs": inputs, "outputs": outputs})
    
    async def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the module with given inputs"""
        prompt = self.signature.to_prompt(inputs)
        
        # Add few-shot examples if available
        if self.examples:
            examples_prompt = "\n\n".join(
                f"Example {i+1}:\nInputs: {json.dumps(ex['inputs'])}\nOutputs: {json.dumps(ex['outputs'])}"
                for i, ex in enumerate(self.examples)
            )
            prompt = f"{examples_prompt}\n\n---\n\n{prompt}"
        
        # Invoke LLM
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        
        # Parse response
        try:
            content = response.content
            # Extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            # Try to find JSON in content
            if "{" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                content = content[json_start:json_end]
            
            return json.loads(content)
        except json.JSONDecodeError:
            # Return raw content if JSON parsing fails
            return {"result": response.content}


class ChainOfThought(DSPyModule):
    """DSPy Chain-of-Thought reasoning module"""
    
    async def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with chain-of-thought reasoning"""
        # Create CoT signature
        cot_signature = DSPySignature(
            name=f"{self.signature.name}_cot",
            inputs=self.signature.inputs,
            outputs={
                "reasoning": "Step-by-step reasoning process",
                **self.signature.outputs
            },
            description=f"{self.signature.description}\n\nThink step-by-step and provide your reasoning before the final answer."
        )
        
        prompt = cot_signature.to_prompt(inputs)
        
        # Add examples with reasoning
        if self.examples:
            examples_prompt = "\n\n".join(
                f"Example {i+1}:\nInputs: {json.dumps(ex['inputs'])}\nReasoning: [Step-by-step thinking]\nOutputs: {json.dumps(ex['outputs'])}"
                for i, ex in enumerate(self.examples)
            )
            prompt = f"{examples_prompt}\n\n---\n\n{prompt}"
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        
        try:
            content = response.content
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            if "{" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                content = content[json_start:json_end]
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {"reasoning": response.content, "result": response.content}


class DSPyProgram:
    """Combines multiple DSPy modules into a pipeline"""
    
    def __init__(self):
        self.modules: Dict[str, DSPyModule] = {}
        self.pipeline: List[str] = []
    
    def add_module(self, name: str, module: DSPyModule) -> 'DSPyProgram':
        """Add a module to the program"""
        self.modules[name] = module
        return self
    
    def set_pipeline(self, module_names: List[str]) -> 'DSPyProgram':
        """Set the execution pipeline"""
        self.pipeline = module_names
        return self
    
    async def execute(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entire pipeline"""
        current_data = {**initial_inputs}
        
        for module_name in self.pipeline:
            if module_name not in self.modules:
                raise ValueError(f"Module '{module_name}' not found in program")
            
            module = self.modules[module_name]
            module_output = await module.forward(current_data)
            current_data = {**current_data, **module_output}
        
        return current_data


class DSPyOptimizer:
    """Optimizes DSPy modules based on training examples"""
    
    def __init__(self, metric: Callable[[Any, Any], float]):
        """
        Args:
            metric: Function that takes (prediction, ground_truth) and returns score (0-1)
        """
        self.metric = metric
    
    async def optimize(
        self,
        module: DSPyModule,
        training_examples: List[Dict[str, Any]],
        validation_split: float = 0.2
    ) -> DSPyModule:
        """Optimize module using training examples"""
        print(f"🔧 Optimizing DSPy module '{module.signature.name}' with {len(training_examples)} examples...")
        
        # Split into train/validation
        split_idx = int(len(training_examples) * (1 - validation_split))
        train_examples = training_examples[:split_idx]
        val_examples = training_examples[split_idx:]
        
        # Add training examples to module
        for example in train_examples:
            module.add_example(example["inputs"], example["outputs"])
        
        # Evaluate on validation set
        if val_examples:
            total_score = 0.0
            for example in val_examples[:5]:  # Evaluate on first 5 validation examples
                prediction = await module.forward(example["inputs"])
                score = self.metric(prediction, example["outputs"])
                total_score += score
            
            avg_score = total_score / min(len(val_examples), 5)
            print(f"✅ Optimization complete. Validation score: {avg_score:.2f}")
        
        return module


# Pre-configured DSPy Modules for Government Recruitment

DocumentAnalysisModule = lambda: DSPyModule(
    signature=DSPySignature(
        name="document_analysis",
        inputs={
            "documentType": "Type of document (AADHAR, PAN, CERTIFICATE, etc.)",
            "extractedText": "Text extracted from document via OCR",
            "candidateData": "Candidate information for verification"
        },
        outputs={
            "isValid": "Boolean indicating if document is valid",
            "confidence": "Confidence score (0-1)",
            "extractedInfo": "Key information extracted from document",
            "issues": "Array of issues found",
            "recommendations": "Array of recommendations"
        },
        description="Analyze and verify government documents for authenticity and completeness"
    ),
    temperature=0.3
)

QuestionGenerationModule = lambda: ChainOfThought(
    signature=DSPySignature(
        name="question_generation",
        inputs={
            "examType": "Type of examination (UPSC, JEE, NEET, etc.)",
            "subject": "Subject area for question",
            "difficulty": "Difficulty level (EASY, MEDIUM, HARD)",
            "syllabus": "Relevant syllabus points",
            "currentAffairs": "Recent news and events (if applicable)"
        },
        outputs={
            "question": "Generated question text",
            "options": "Array of 4 options for MCQ",
            "correctAnswer": "Correct answer",
            "explanation": "Detailed explanation",
            "tags": "Array of relevant tags",
            "bloomsLevel": "Bloom's taxonomy level"
        },
        description="Generate unique, high-quality examination questions ensuring zero repetition"
    ),
    temperature=0.8
)

RankingCalculationModule = lambda: ChainOfThought(
    signature=DSPySignature(
        name="ranking_calculation",
        inputs={
            "candidateScores": "Array of candidate scores with categories",
            "examType": "Type of examination",
            "cutoffCriteria": "Cutoff calculation criteria",
            "reservationPolicy": "Reservation policy details"
        },
        outputs={
            "rankings": "Overall and category-wise rankings",
            "cutoffs": "Calculated cutoff marks for each category",
            "qualifiedCandidates": "List of qualified candidates",
            "statistics": "Statistical analysis of results"
        },
        description="Calculate rankings, cutoffs, and merit lists following government reservation policies"
    ),
    temperature=0.2
)

EligibilityCheckModule = lambda: DSPyModule(
    signature=DSPySignature(
        name="eligibility_check",
        inputs={
            "candidateAge": "Candidate age",
            "category": "Candidate category (GENERAL, OBC, SC, ST, EWS, PWD)",
            "education": "Educational qualifications",
            "examCriteria": "Examination eligibility criteria"
        },
        outputs={
            "isEligible": "Boolean indicating eligibility",
            "reasons": "Array of reasons for eligibility/ineligibility",
            "relaxations": "Applicable age/qualification relaxations"
        },
        description="Verify candidate eligibility for government examinations"
    ),
    temperature=0.1
)

ResumeAnalysisModule = lambda: ChainOfThought(
    signature=DSPySignature(
        name="resume_analysis",
        inputs={
            "resumeText": "Extracted text from resume",
            "jobDescription": "Job requirements and description",
            "requiredSkills": "Required skills list",
            "experienceLevel": "Required experience level"
        },
        outputs={
            "matchScore": "Match score (0-100)",
            "extractedInfo": "Candidate information (name, email, phone, skills, experience)",
            "strengths": "List of candidate strengths",
            "weaknesses": "List of candidate weaknesses",
            "recommendations": "Hiring recommendations",
            "redFlags": "Any red flags detected"
        },
        description="Analyze resume and match candidate to job requirements"
    ),
    temperature=0.3
)

InterviewEvaluationModule = lambda: ChainOfThought(
    signature=DSPySignature(
        name="interview_evaluation",
        inputs={
            "questions": "List of interview questions",
            "answers": "List of candidate answers",
            "jobRole": "Job role being interviewed for",
            "proctoringData": "Proctoring and behavioral data"
        },
        outputs={
            "overallScore": "Overall interview score (0-100)",
            "technicalSkills": "Technical skills scores",
            "softSkills": "Soft skills scores",
            "strengths": "Candidate strengths",
            "areasOfImprovement": "Areas needing improvement",
            "recommendation": "Hiring recommendation (STRONG_YES, YES, MAYBE, NO)",
            "detailedFeedback": "Detailed feedback for candidate"
        },
        description="Evaluate interview responses and provide comprehensive assessment"
    ),
    temperature=0.2
)


# Pre-configured Programs

def create_application_processing_program() -> DSPyProgram:
    """Create program for complete application processing"""
    program = DSPyProgram()
    program.add_module("eligibility", EligibilityCheckModule())
    program.add_module("documentAnalysis", DocumentAnalysisModule())
    program.set_pipeline(["eligibility", "documentAnalysis"])
    return program


def create_assessment_generation_program() -> DSPyProgram:
    """Create program for assessment question generation"""
    program = DSPyProgram()
    program.add_module("questionGeneration", QuestionGenerationModule())
    program.set_pipeline(["questionGeneration"])
    return program


def create_resume_screening_program() -> DSPyProgram:
    """Create program for resume screening"""
    program = DSPyProgram()
    program.add_module("resumeAnalysis", ResumeAnalysisModule())
    program.set_pipeline(["resumeAnalysis"])
    return program


def create_interview_evaluation_program() -> DSPyProgram:
    """Create program for interview evaluation"""
    program = DSPyProgram()
    program.add_module("interviewEvaluation", InterviewEvaluationModule())
    program.set_pipeline(["interviewEvaluation"])
    return program