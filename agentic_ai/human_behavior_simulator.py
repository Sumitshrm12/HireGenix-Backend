"""
🎭 HUMAN BEHAVIOR SIMULATOR - Making AI Interviews Indistinguishable from Human
Implements natural conversational patterns, micro-acknowledgments, thinking pauses,
and empathetic responses that mirror human interviewer behavior.

Features:
- Natural micro-acknowledgments ("I see", "That's interesting", "Hmm")
- Thinking pause simulation with filler phrases
- Empathetic response generation
- Active listening indicators
- Context-aware transitions
- Personal anecdotes (persona-based)
- Clarification seeking behavior
- Natural conversation flow
- Interruption handling
- Turn-taking management

Tech Stack:
- LangGraph for conversation flow management
- Azure OpenAI for dynamic phrase generation
- State machine for conversation state tracking
"""

import os
import json
import random
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationMoment(str, Enum):
    """Types of conversation moments requiring human behavior"""
    ACKNOWLEDGMENT = "acknowledgment"
    THINKING_PAUSE = "thinking_pause"
    ENCOURAGEMENT = "encouragement"
    CLARIFICATION_REQUEST = "clarification_request"
    TOPIC_TRANSITION = "topic_transition"
    ACTIVE_LISTENING = "active_listening"
    EMPATHETIC_RESPONSE = "empathetic_response"
    POSITIVE_FEEDBACK = "positive_feedback"
    GENTLE_CHALLENGE = "gentle_challenge"
    CONVERSATION_REPAIR = "conversation_repair"
    INTERVIEW_OPENING = "interview_opening"
    INTERVIEW_CLOSING = "interview_closing"


class InterviewerPersona(str, Enum):
    """Interviewer personality types"""
    WARM_MENTOR = "warm_mentor"
    TECHNICAL_PEER = "technical_peer"
    EXECUTIVE_LEADER = "executive_leader"
    HR_PROFESSIONAL = "hr_professional"
    STARTUP_ENTHUSIAST = "startup_enthusiast"


@dataclass
class PersonaProfile:
    """Profile for an interviewer persona"""
    name: str
    title: str
    communication_style: str
    typical_phrases: List[str]
    anecdotes: List[str]
    values_emphasized: List[str]
    question_style: str
    feedback_style: str


class HumanBehaviorSimulator:
    """
    Simulates natural human interviewer behavior to create authentic
    interview experiences indistinguishable from human interviewers.
    """
    
    def __init__(self):
        logger.info("🎭 Initializing Human Behavior Simulator...")
        
        # Azure OpenAI for dynamic generation
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.7
        )
        
        # Initialize persona profiles
        self.personas = self._init_personas()
        
        # Initialize phrase libraries
        self._init_phrase_libraries()
        
        logger.info("✅ Human Behavior Simulator initialized")
    
    def _init_personas(self) -> Dict[InterviewerPersona, PersonaProfile]:
        """Initialize interviewer persona profiles"""
        return {
            InterviewerPersona.WARM_MENTOR: PersonaProfile(
                name="Sarah",
                title="Senior Engineering Manager",
                communication_style="warm, encouraging, supportive",
                typical_phrases=[
                    "That's a great point.",
                    "I really appreciate you sharing that.",
                    "You know, that reminds me of...",
                    "I can tell you've put a lot of thought into this."
                ],
                anecdotes=[
                    "When I was earlier in my career, I faced something similar...",
                    "I remember working on a project just like that...",
                    "One of my mentors once told me something that stuck with me..."
                ],
                values_emphasized=["growth", "learning", "collaboration", "mentorship"],
                question_style="open-ended, exploratory",
                feedback_style="constructive, growth-focused"
            ),
            InterviewerPersona.TECHNICAL_PEER: PersonaProfile(
                name="Alex",
                title="Principal Engineer",
                communication_style="direct, technical, curious",
                typical_phrases=[
                    "Interesting approach.",
                    "Let me dig into that a bit more.",
                    "How would you handle edge cases here?",
                    "That's a solid architectural decision."
                ],
                anecdotes=[
                    "We actually ran into that exact problem last quarter...",
                    "I've seen teams go both ways on this...",
                    "There's an interesting trade-off there that we debated..."
                ],
                values_emphasized=["technical excellence", "scalability", "code quality"],
                question_style="probing, specific",
                feedback_style="technical, detailed"
            ),
            InterviewerPersona.EXECUTIVE_LEADER: PersonaProfile(
                name="Michael",
                title="VP of Engineering",
                communication_style="strategic, big-picture, decisive",
                typical_phrases=[
                    "From a business perspective...",
                    "How does this scale?",
                    "What's the impact on the bottom line?",
                    "Tell me about the strategic thinking here."
                ],
                anecdotes=[
                    "When we were scaling from 10 to 100 engineers...",
                    "I've seen this pattern succeed when...",
                    "The best teams I've led have always..."
                ],
                values_emphasized=["impact", "leadership", "vision", "execution"],
                question_style="strategic, outcome-focused",
                feedback_style="direct, actionable"
            ),
            InterviewerPersona.HR_PROFESSIONAL: PersonaProfile(
                name="Jennifer",
                title="Head of Talent",
                communication_style="empathetic, relationship-focused, inclusive",
                typical_phrases=[
                    "I'd love to understand more about...",
                    "How did that experience shape you?",
                    "What energizes you about this kind of work?",
                    "That shows real emotional intelligence."
                ],
                anecdotes=[
                    "Some of our best team members came from similar backgrounds...",
                    "What I've noticed about successful people here...",
                    "The culture we're building values..."
                ],
                values_emphasized=["culture fit", "values alignment", "team dynamics"],
                question_style="behavioral, values-based",
                feedback_style="supportive, holistic"
            ),
            InterviewerPersona.STARTUP_ENTHUSIAST: PersonaProfile(
                name="Jordan",
                title="Co-founder & CTO",
                communication_style="energetic, informal, fast-paced",
                typical_phrases=[
                    "That's awesome!",
                    "Move fast and iterate, right?",
                    "What would you build first?",
                    "I love that scrappy approach."
                ],
                anecdotes=[
                    "When we were just three people in a garage...",
                    "We shipped that feature in a weekend once...",
                    "The best ideas come from anywhere..."
                ],
                values_emphasized=["agility", "ownership", "innovation", "hustle"],
                question_style="hands-on, practical",
                feedback_style="enthusiastic, action-oriented"
            )
        }
    
    def _init_phrase_libraries(self):
        """Initialize comprehensive phrase libraries for human-like behavior"""
        
        # Micro-acknowledgments (brief responses during listening)
        self.micro_acknowledgments = {
            "neutral": [
                "I see.",
                "Mm-hmm.",
                "Right.",
                "Okay.",
                "Got it.",
                "Understood.",
            ],
            "interested": [
                "Interesting.",
                "That's fascinating.",
                "Tell me more.",
                "Oh, really?",
                "That's a unique perspective.",
            ],
            "impressed": [
                "Impressive.",
                "Nice.",
                "Solid.",
                "That's excellent.",
                "Very thorough.",
            ],
            "thoughtful": [
                "Hmm, let me think about that.",
                "That's thought-provoking.",
                "I hadn't considered that angle.",
                "That's an interesting way to look at it.",
            ]
        }
        
        # Thinking pause fillers
        self.thinking_fillers = [
            "That's a great point. Let me think about how to frame my next question...",
            "Hmm, that's interesting. I want to dig into something you mentioned...",
            "So, based on what you've shared...",
            "You know, that actually brings up an important point...",
            "Let me see... There's something I want to explore further here...",
            "That's helpful context. Now I'm curious about...",
            "Okay, processing all of that...",
        ]
        
        # Active listening indicators
        self.active_listening = {
            "reflection": [
                "So what I'm hearing is...",
                "If I understand correctly...",
                "So you're saying that...",
                "It sounds like...",
            ],
            "summary": [
                "To summarize what you've shared...",
                "So the key points are...",
                "Let me make sure I've got this right...",
            ],
            "connection": [
                "That connects to what you mentioned earlier about...",
                "I see how that relates to...",
                "That's consistent with your point about...",
            ]
        }
        
        # Empathetic responses
        self.empathetic_responses = {
            "struggle_acknowledged": [
                "That sounds like it was challenging.",
                "I can imagine that was difficult.",
                "That's a tough situation to navigate.",
                "It takes courage to share that.",
            ],
            "success_celebrated": [
                "That's a fantastic achievement!",
                "You should be proud of that.",
                "That's exactly the kind of impact we look for.",
                "Impressive work on that.",
            ],
            "frustration_validated": [
                "I can understand why that would be frustrating.",
                "That's a common challenge, actually.",
                "Many people have struggled with similar situations.",
            ],
            "excitement_matched": [
                "I can hear the passion in what you're describing!",
                "It's clear you really care about this.",
                "That enthusiasm is contagious!",
            ]
        }
        
        # Clarification requests
        self.clarification_requests = [
            "Could you elaborate on that a bit?",
            "Can you give me a specific example?",
            "What do you mean by...?",
            "Could you walk me through that in more detail?",
            "I want to make sure I understand - are you saying...?",
            "That's interesting. Can you unpack that a bit more?",
            "Help me understand the context there.",
        ]
        
        # Topic transitions
        self.topic_transitions = {
            "smooth": [
                "That's really helpful. I'd like to shift gears a bit and talk about...",
                "Thank you for that insight. Let's explore another area...",
                "Great. Now I'm curious about a different aspect of your experience...",
            ],
            "connected": [
                "Building on what you just shared, I'd like to ask about...",
                "That actually leads nicely into my next question about...",
                "Speaking of which, how do you approach...",
            ],
            "direct": [
                "Let's move on to discuss...",
                "Now I'd like to talk about...",
                "Shifting focus here...",
            ]
        }
        
        # Conversation repair phrases
        self.conversation_repair = {
            "misunderstanding": [
                "I may not have been clear. Let me try again...",
                "Actually, I think I phrased that poorly. What I meant was...",
                "Let me rephrase that question...",
            ],
            "technical_issue": [
                "Sorry, I think we may have had a connection issue. Could you repeat that?",
                "I want to make sure I caught everything. Can you say that again?",
            ],
            "off_track": [
                "Let's bring it back to the main topic.",
                "That's great context, but I'm particularly interested in...",
                "Good background. Specifically though, I'm wondering about...",
            ]
        }
        
        # Opening phrases by persona
        self.interview_openings = {
            InterviewerPersona.WARM_MENTOR: [
                "Hi {name}! I'm Sarah, and I'm really excited to chat with you today. Before we dive in, how are you doing? Did you find everything okay?",
                "Hello {name}! Welcome! I'm Sarah from the engineering team. First of all, thank you for taking the time to speak with us. How's your day going?",
            ],
            InterviewerPersona.TECHNICAL_PEER: [
                "Hey {name}, I'm Alex. I'll be leading the technical portion of our chat today. Ready to geek out a bit?",
                "Hi {name}! Alex here. I'm one of the principal engineers. Looking forward to digging into some technical stuff with you today.",
            ],
            InterviewerPersona.EXECUTIVE_LEADER: [
                "Hello {name}, I'm Michael, VP of Engineering. Thanks for joining me today. I've heard great things from the earlier rounds.",
                "{name}, welcome. I'm Michael. I lead our engineering organization. Let's have a conversation about where you see yourself making an impact.",
            ],
            InterviewerPersona.HR_PROFESSIONAL: [
                "Hi {name}! I'm Jennifer from our Talent team. I'm thrilled to meet you today. Let's have a real conversation - this isn't an interrogation!",
                "Hello {name}! Welcome! I'm Jennifer, and I'm here to learn more about you as a person, not just your resume. Shall we begin?",
            ],
            InterviewerPersona.STARTUP_ENTHUSIAST: [
                "Hey {name}! Jordan here, CTO and co-founder. Super pumped to chat with you! Let's skip the formalities - tell me what you're excited about!",
                "{name}! Great to meet you. I'm Jordan. We move fast here, so let's dive right in. What gets you fired up about this opportunity?",
            ]
        }
        
        # Closing phrases by persona
        self.interview_closings = {
            InterviewerPersona.WARM_MENTOR: [
                "This has been a wonderful conversation, {name}. I really enjoyed learning about your journey. Do you have any questions for me?",
                "Thank you so much for sharing, {name}. I can tell you've put real thought into your career. What questions do you have about the team or role?",
            ],
            InterviewerPersona.TECHNICAL_PEER: [
                "Good stuff, {name}. I enjoyed the technical discussion. What questions do you have about the tech stack or how we work?",
                "That was a solid chat. Any questions for me about the engineering culture or the challenges we're tackling?",
            ],
            InterviewerPersona.EXECUTIVE_LEADER: [
                "{name}, I appreciate the strategic thinking you've shown today. What questions do you have about our vision or where the company is headed?",
                "Great conversation. Before we wrap up, what do you want to know about leadership here or the company's direction?",
            ],
            InterviewerPersona.HR_PROFESSIONAL: [
                "{name}, thank you for being so open with me today. What questions do you have about our culture, benefits, or anything else?",
                "I've really enjoyed getting to know you. What's on your mind? Any questions about what it's like to work here?",
            ],
            InterviewerPersona.STARTUP_ENTHUSIAST: [
                "Awesome chat, {name}! I think you'd fit right in. What do you want to know about the startup life or what we're building?",
                "Love the energy! Before we wrap, what's burning? Any questions about the product, team, or how crazy it gets around here?",
            ]
        }
    
    def get_micro_acknowledgment(
        self,
        context: str = "neutral",
        persona: Optional[InterviewerPersona] = None
    ) -> str:
        """Get a contextually appropriate micro-acknowledgment"""
        base_phrases = self.micro_acknowledgments.get(context, self.micro_acknowledgments["neutral"])
        selected = random.choice(base_phrases)
        
        # Add persona flavor if provided
        if persona and persona in self.personas:
            persona_profile = self.personas[persona]
            if random.random() < 0.3:  # 30% chance to use persona-specific phrase
                selected = random.choice(persona_profile.typical_phrases[:2])
        
        return selected
    
    def get_thinking_pause(
        self,
        context: Optional[str] = None,
        persona: Optional[InterviewerPersona] = None
    ) -> Dict[str, Any]:
        """Get a thinking pause with filler phrase and suggested delay"""
        filler = random.choice(self.thinking_fillers)
        
        # Adjust delay based on context
        base_delay = random.uniform(1.5, 3.5)
        
        return {
            "filler_phrase": filler,
            "suggested_delay_seconds": base_delay,
            "display_typing": True
        }
    
    def get_active_listening_phrase(
        self,
        listening_type: str,
        referenced_content: Optional[str] = None
    ) -> str:
        """Get an active listening phrase"""
        phrases = self.active_listening.get(listening_type, self.active_listening["reflection"])
        selected = random.choice(phrases)
        
        if referenced_content:
            selected = f"{selected} {referenced_content}"
        
        return selected
    
    def get_empathetic_response(
        self,
        emotion_detected: str,
        persona: Optional[InterviewerPersona] = None
    ) -> str:
        """Get an empathetic response based on detected emotion"""
        emotion_map = {
            "struggling": "struggle_acknowledged",
            "proud": "success_celebrated",
            "frustrated": "frustration_validated",
            "excited": "excitement_matched",
            "nervous": "struggle_acknowledged",
            "confident": "success_celebrated"
        }
        
        response_type = emotion_map.get(emotion_detected, "struggle_acknowledged")
        responses = self.empathetic_responses.get(response_type, [])
        
        if responses:
            return random.choice(responses)
        return "I appreciate you sharing that."
    
    def get_clarification_request(
        self,
        topic: Optional[str] = None,
        style: str = "gentle"
    ) -> str:
        """Get a clarification request phrase"""
        request = random.choice(self.clarification_requests)
        
        if topic:
            # Add topic-specific context
            request = request.replace("that", f"your experience with {topic}")
        
        return request
    
    def get_topic_transition(
        self,
        transition_style: str = "smooth",
        next_topic: Optional[str] = None
    ) -> str:
        """Get a topic transition phrase"""
        transitions = self.topic_transitions.get(transition_style, self.topic_transitions["smooth"])
        transition = random.choice(transitions)
        
        if next_topic:
            transition = f"{transition} {next_topic}"
        
        return transition
    
    def get_conversation_repair(
        self,
        repair_type: str,
        context: Optional[str] = None
    ) -> str:
        """Get a conversation repair phrase"""
        repairs = self.conversation_repair.get(repair_type, self.conversation_repair["misunderstanding"])
        return random.choice(repairs)
    
    def get_interview_opening(
        self,
        persona: InterviewerPersona,
        candidate_name: str
    ) -> str:
        """Get a personalized interview opening"""
        openings = self.interview_openings.get(
            persona,
            self.interview_openings[InterviewerPersona.WARM_MENTOR]
        )
        opening = random.choice(openings)
        return opening.format(name=candidate_name)
    
    def get_interview_closing(
        self,
        persona: InterviewerPersona,
        candidate_name: str
    ) -> str:
        """Get a personalized interview closing"""
        closings = self.interview_closings.get(
            persona,
            self.interview_closings[InterviewerPersona.WARM_MENTOR]
        )
        closing = random.choice(closings)
        return closing.format(name=candidate_name)
    
    async def generate_contextual_response(
        self,
        moment_type: ConversationMoment,
        context: Dict[str, Any],
        persona: InterviewerPersona = InterviewerPersona.WARM_MENTOR
    ) -> Dict[str, Any]:
        """Generate a contextually appropriate human-like response"""
        persona_profile = self.personas[persona]
        
        prompt = f"""
        You are {persona_profile.name}, a {persona_profile.title}.
        Communication style: {persona_profile.communication_style}
        
        CONTEXT: {json.dumps(context)}
        
        Generate a natural, human-like response for this moment: {moment_type.value}
        
        The response should:
        1. Sound completely natural and conversational
        2. Match the persona's communication style
        3. Be appropriate for a professional interview setting
        4. Be concise (1-2 sentences max)
        
        Return JSON:
        {{
            "response": "The natural response",
            "tone": "warm/professional/curious/encouraging",
            "suggested_delay_seconds": 1.5
        }}
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            return {
                "success": True,
                "moment_type": moment_type.value,
                "response": result.get("response", ""),
                "tone": result.get("tone", "neutral"),
                "suggested_delay_seconds": result.get("suggested_delay_seconds", 1.5),
                "persona": persona.value
            }
        except Exception as e:
            logger.error(f"❌ Contextual response generation error: {e}")
            # Fallback to library phrases
            return {
                "success": False,
                "moment_type": moment_type.value,
                "response": self.get_micro_acknowledgment("neutral", persona),
                "tone": "neutral",
                "suggested_delay_seconds": 1.0,
                "persona": persona.value
            }
    
    async def humanize_question(
        self,
        base_question: str,
        context: Dict[str, Any],
        persona: InterviewerPersona = InterviewerPersona.WARM_MENTOR,
        include_acknowledgment: bool = True,
        include_transition: bool = False
    ) -> Dict[str, Any]:
        """Transform a base question into a natural, human-like question"""
        persona_profile = self.personas[persona]
        
        # Build the humanized question
        components = []
        
        # Add acknowledgment if there was a previous answer
        if include_acknowledgment and context.get("previous_answer"):
            ack = self.get_micro_acknowledgment("interested", persona)
            components.append(ack)
        
        # Add transition if switching topics
        if include_transition:
            transition = self.get_topic_transition("connected")
            components.append(transition)
        
        # Generate humanized version of the question
        prompt = f"""
        You are {persona_profile.name}, a {persona_profile.title}.
        Communication style: {persona_profile.communication_style}
        
        BASE QUESTION: {base_question}
        
        Transform this into a natural, conversational question that:
        1. Sounds like a human is speaking, not reading from a script
        2. Matches the persona's style
        3. May include brief context or rationale for asking
        4. Is warm and inviting
        
        Previous acknowledgment/transition already added: {' '.join(components) if components else 'None'}
        
        Return ONLY the transformed question text, no JSON or extra formatting.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            humanized = response.content.strip()
            
            # Combine with acknowledgment/transition
            if components:
                full_response = " ".join(components) + " " + humanized
            else:
                full_response = humanized
            
            return {
                "success": True,
                "original_question": base_question,
                "humanized_question": full_response,
                "components": {
                    "acknowledgment": components[0] if include_acknowledgment and components else None,
                    "transition": components[-1] if include_transition and len(components) > 1 else None,
                    "question": humanized
                },
                "persona": persona.value,
                "suggested_delivery": {
                    "pause_before_question": random.uniform(0.5, 1.5),
                    "speaking_pace": "moderate"
                }
            }
        except Exception as e:
            logger.error(f"❌ Question humanization error: {e}")
            return {
                "success": False,
                "original_question": base_question,
                "humanized_question": base_question,
                "error": str(e)
            }
    
    def get_persona_anecdote(
        self,
        persona: InterviewerPersona,
        topic: Optional[str] = None
    ) -> str:
        """Get a persona-appropriate anecdote for rapport building"""
        profile = self.personas.get(persona, self.personas[InterviewerPersona.WARM_MENTOR])
        return random.choice(profile.anecdotes)


# Singleton instance
_human_simulator = None

def get_human_behavior_simulator() -> HumanBehaviorSimulator:
    """Get singleton human behavior simulator instance"""
    global _human_simulator
    if _human_simulator is None:
        _human_simulator = HumanBehaviorSimulator()
    return _human_simulator
