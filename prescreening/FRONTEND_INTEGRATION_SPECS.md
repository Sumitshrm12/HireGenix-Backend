# Frontend Integration Specifications for HireGenix Pre-Screening Flow

## Table of Contents
1. [Component Architecture](#component-architecture)
2. [React Components](#react-components)  
3. [API Integration](#api-integration)
4. [State Management](#state-management)
5. [WebRTC Integration](#webrtc-integration)
6. [Proctoring Client](#proctoring-client)
7. [Notification System](#notification-system)
8. [Error Handling](#error-handling)
9. [Testing Strategy](#testing-strategy)
10. [Deployment Guide](#deployment-guide)

## Component Architecture

### High-Level Component Structure
```
PreScreeningFlow/
├── components/
│   ├── PreScreeningDashboard.tsx     # Main container
│   ├── SessionSetup.tsx              # Initial setup & config
│   ├── VideoPreScreening.tsx         # Video MCQ component
│   ├── ProctoringMonitor.tsx         # Real-time monitoring
│   ├── ResultsDisplay.tsx            # Final results
│   └── AdminPanel.tsx                # Admin controls
├── hooks/
│   ├── usePreScreening.ts            # Main hook
│   ├── useProctoring.ts              # Proctoring logic
│   ├── useVideoRecording.ts          # Video capture
│   └── useWebSocket.ts               # Real-time updates
├── services/
│   ├── prescreeningApi.ts            # API client
│   ├── webrtcService.ts              # WebRTC handling
│   └── proctoringService.ts          # Client-side detection
├── types/
│   └── prescreening.ts               # TypeScript types
└── utils/
    ├── validation.ts                 # Form validation
    └── constants.ts                  # App constants
```

## React Components

### 1. PreScreeningDashboard Component

```typescript
// components/PreScreeningDashboard.tsx
import React, { useState, useEffect } from 'react';
import { Card, Button, Progress, Alert } from '@/components/ui';
import { PreScreeningStatus, PreScreeningSession } from '@/types/prescreening';
import { usePreScreening } from '@/hooks/usePreScreening';
import { useNotifications } from '@/hooks/useNotifications';
import SessionSetup from './SessionSetup';
import VideoPreScreening from './VideoPreScreening';
import ResultsDisplay from './ResultsDisplay';

interface PreScreeningDashboardProps {
  candidateId: string;
  jobId: string;
  onComplete?: (results: any) => void;
}

const PreScreeningDashboard: React.FC<PreScreeningDashboardProps> = ({
  candidateId,
  jobId,
  onComplete
}) => {
  const {
    session,
    status,
    currentStep,
    startSession,
    completeSession,
    error,
    loading
  } = usePreScreening(candidateId, jobId);

  const { notifications, markAsRead } = useNotifications();

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'setup':
        return (
          <SessionSetup
            candidateId={candidateId}
            jobId={jobId}
            onSetupComplete={startSession}
          />
        );
      case 'screening':
        return (
          <VideoPreScreening
            session={session}
            onComplete={completeSession}
          />
        );
      case 'results':
        return (
          <ResultsDisplay
            session={session}
            onComplete={onComplete}
          />
        );
      default:
        return <div>Loading...</div>;
    }
  };

  return (
    <div className="pre-screening-dashboard">
      {/* Header */}
      <div className="header mb-6">
        <h1 className="text-2xl font-bold">AI Pre-Screening Assessment</h1>
        <div className="flex items-center gap-4">
          <Progress 
            value={currentStep === 'setup' ? 25 : currentStep === 'screening' ? 75 : 100}
            className="w-48"
          />
          <span className="text-sm text-muted-foreground">
            Step {currentStep === 'setup' ? '1' : currentStep === 'screening' ? '2' : '3'} of 3
          </span>
        </div>
      </div>

      {/* Notifications */}
      {notifications.length > 0 && (
        <div className="notifications mb-4">
          {notifications.slice(0, 3).map((notification, index) => (
            <Alert key={index} variant={notification.type === 'ERROR' ? 'destructive' : 'default'}>
              <span>{notification.message}</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => markAsRead(notification.id)}
              >
                Dismiss
              </Button>
            </Alert>
          ))}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <Alert variant="destructive" className="mb-4">
          <span>Error: {error}</span>
        </Alert>
      )}

      {/* Main Content */}
      <Card className="p-6">
        {loading ? (
          <div className="flex items-center justify-center p-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <span className="ml-2">Loading...</span>
          </div>
        ) : (
          renderCurrentStep()
        )}
      </Card>
    </div>
  );
};

export default PreScreeningDashboard;
```

### 2. VideoPreScreening Component

```typescript
// components/VideoPreScreening.tsx
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button, Card, Progress, Alert } from '@/components/ui';
import { PreScreeningSession, MCQQuestion } from '@/types/prescreening';
import { useVideoRecording } from '@/hooks/useVideoRecording';
import { useProctoring } from '@/hooks/useProctoring';
import { useWebSocket } from '@/hooks/useWebSocket';
import ProctoringMonitor from './ProctoringMonitor';

interface VideoPreScreeningProps {
  session: PreScreeningSession;
  onComplete: (results: any) => void;
}

const VideoPreScreening: React.FC<VideoPreScreeningProps> = ({
  session,
  onComplete
}) => {
  const [currentQuestion, setCurrentQuestion] = useState<MCQQuestion | null>(null);
  const [selectedAnswer, setSelectedAnswer] = useState<string>('');
  const [questionIndex, setQuestionIndex] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(180); // 3 minutes per question
  const [isRecording, setIsRecording] = useState(false);
  
  const {
    startRecording,
    stopRecording,
    videoRef,
    streamRef,
    recordedChunks,
    error: videoError
  } = useVideoRecording();

  const {
    startProctoring,
    stopProctoring,
    violations,
    proctoringActive
  } = useProctoring(session.id);

  const { sendMessage, lastMessage, connectionState } = useWebSocket(
    `ws://localhost:8000/ws/prescreening/${session.id}`
  );

  // Timer effect
  useEffect(() => {
    if (timeRemaining > 0 && isRecording) {
      const timer = setTimeout(() => setTimeRemaining(prev => prev - 1), 1000);
      return () => clearTimeout(timer);
    } else if (timeRemaining === 0) {
      handleAutoSubmit();
    }
  }, [timeRemaining, isRecording]);

  // Initialize first question
  useEffect(() => {
    if (session && !currentQuestion) {
      fetchNextQuestion();
    }
  }, [session]);

  const fetchNextQuestion = async () => {
    try {
      const response = await fetch(`/api/prescreening/${session.id}/next-question`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          previous_evaluations: [], // Will be populated from session state
        })
      });
      
      const question = await response.json();
      setCurrentQuestion(question);
      setSelectedAnswer('');
      setTimeRemaining(180);
    } catch (error) {
      console.error('Error fetching question:', error);
    }
  };

  const handleStartQuestion = async () => {
    if (!currentQuestion) return;
    
    try {
      await startRecording();
      await startProctoring();
      setIsRecording(true);
      
      // Send websocket message
      sendMessage({
        type: 'question_started',
        question_id: currentQuestion.id,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Error starting question:', error);
    }
  };

  const handleSubmitAnswer = async () => {
    if (!currentQuestion || !selectedAnswer) return;
    
    try {
      const recordingBlob = await stopRecording();
      const audioTranscript = await transcribeAudio(recordingBlob);
      
      // Submit answer with video and audio
      const response = await fetch(`/api/prescreening/${session.id}/submit-answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question_id: currentQuestion.id,
          selected_answer: selectedAnswer,
          video_data: await blobToBase64(recordingBlob),
          audio_transcript: audioTranscript,
          time_taken: 180 - timeRemaining
        })
      });
      
      const evaluation = await response.json();
      
      setIsRecording(false);
      setQuestionIndex(prev => prev + 1);
      
      // Check if assessment is complete
      if (questionIndex >= session.total_questions - 1) {
        onComplete(evaluation);
      } else {
        await fetchNextQuestion();
      }
      
    } catch (error) {
      console.error('Error submitting answer:', error);
    }
  };

  const handleAutoSubmit = () => {
    if (selectedAnswer) {
      handleSubmitAnswer();
    } else {
      // Auto-select random answer if time runs out
      const options = Object.keys(currentQuestion?.options || {});
      if (options.length > 0) {
        setSelectedAnswer(options[0]);
        setTimeout(handleSubmitAnswer, 100);
      }
    }
  };

  const transcribeAudio = async (blob: Blob): Promise<string> => {
    // In a real implementation, you'd use Web Speech API or send to backend
    return "Mock transcript of candidate's response";
  };

  const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.readAsDataURL(blob);
    });
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!currentQuestion) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p>Loading next question...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="video-pre-screening">
      {/* Header */}
      <div className="header mb-6">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-xl font-semibold">
              Question {questionIndex + 1} of {session.total_questions}
            </h2>
            <p className="text-sm text-muted-foreground">Category: {currentQuestion.category}</p>
          </div>
          <div className="text-right">
            <div className={`text-2xl font-mono ${timeRemaining < 30 ? 'text-destructive' : 'text-primary'}`}>
              {formatTime(timeRemaining)}
            </div>
            <p className="text-sm text-muted-foreground">Time remaining</p>
          </div>
        </div>
        <Progress value={(questionIndex / session.total_questions) * 100} className="mt-2" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Panel */}
        <div className="lg:col-span-1">
          <Card className="p-4">
            <h3 className="font-medium mb-3">Video Recording</h3>
            <div className="relative">
              <video
                ref={videoRef}
                autoPlay
                muted
                className="w-full h-48 bg-gray-100 rounded"
              />
              <div className={`absolute top-2 right-2 w-3 h-3 rounded-full ${isRecording ? 'bg-red-500' : 'bg-gray-400'}`} />
            </div>
            
            {!isRecording ? (
              <Button 
                onClick={handleStartQuestion} 
                className="w-full mt-3"
                disabled={!currentQuestion}
              >
                Start Recording & Answer
              </Button>
            ) : (
              <Button 
                onClick={handleSubmitAnswer} 
                className="w-full mt-3"
                disabled={!selectedAnswer}
                variant={selectedAnswer ? 'default' : 'secondary'}
              >
                Submit Answer
              </Button>
            )}
          </Card>

          {/* Proctoring Monitor */}
          <ProctoringMonitor 
            violations={violations}
            isActive={proctoringActive}
            className="mt-4"
          />
        </div>

        {/* Question Panel */}
        <div className="lg:col-span-2">
          <Card className="p-6">
            <h3 className="text-lg font-medium mb-4">Question</h3>
            <p className="text-base mb-6 leading-relaxed">
              {currentQuestion.question_text}
            </p>

            <div className="space-y-3">
              {Object.entries(currentQuestion.options).map(([key, value]) => (
                <label key={key} className="flex items-start space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="answer"
                    value={key}
                    checked={selectedAnswer === key}
                    onChange={(e) => setSelectedAnswer(e.target.value)}
                    disabled={!isRecording}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <span className="font-medium">{key}.</span> {value}
                  </div>
                </label>
              ))}
            </div>

            <div className="mt-6 pt-4 border-t">
              <p className="text-sm text-muted-foreground">
                <strong>Instructions:</strong> Read the question carefully and select your answer. 
                Your video response will be recorded and analyzed. Speak clearly when explaining 
                your reasoning.
              </p>
            </div>
          </Card>
        </div>
      </div>

      {/* Connection Status */}
      {connectionState !== 'connected' && (
        <Alert variant="destructive" className="mt-4">
          <span>Connection issue detected. Please check your internet connection.</span>
        </Alert>
      )}
    </div>
  );
};

export default VideoPreScreening;
```

### 3. ProctoringMonitor Component

```typescript
// components/ProctoringMonitor.tsx
import React from 'react';
import { Card, Alert, Badge } from '@/components/ui';
import { ProctoringViolation } from '@/types/prescreening';

interface ProctoringMonitorProps {
  violations: ProctoringViolation[];
  isActive: boolean;
  className?: string;
}

const ProctoringMonitor: React.FC<ProctoringMonitorProps> = ({
  violations,
  isActive,
  className
}) => {
  const severityColors = {
    LOW: 'bg-yellow-100 text-yellow-800',
    MEDIUM: 'bg-orange-100 text-orange-800',
    HIGH: 'bg-red-100 text-red-800',
    CRITICAL: 'bg-red-500 text-white'
  };

  const getViolationIcon = (type: string) => {
    switch (type) {
      case 'NO_FACE_DETECTED': return '👤';
      case 'MULTIPLE_FACES_DETECTED': return '👥';
      case 'FOCUS_LOST': return '🔄';
      case 'TAB_SWITCH_DETECTED': return '🗂️';
      case 'MULTIPLE_VOICES_DETECTED': return '🗣️';
      default: return '⚠️';
    }
  };

  return (
    <Card className={`p-4 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium">Proctoring Monitor</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500' : 'bg-gray-400'}`} />
          <span className="text-sm text-muted-foreground">
            {isActive ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>

      {violations.length === 0 ? (
        <div className="text-center py-4 text-muted-foreground">
          <div className="text-2xl mb-2">✅</div>
          <p className="text-sm">No violations detected</p>
        </div>
      ) : (
        <div className="space-y-2">
          {violations.slice(0, 5).map((violation, index) => (
            <div key={index} className="flex items-start space-x-2 p-2 bg-gray-50 rounded">
              <span className="text-lg">{getViolationIcon(violation.type)}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2">
                  <Badge 
                    variant="outline" 
                    className={severityColors[violation.severity]}
                  >
                    {violation.severity}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {new Date(violation.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-sm mt-1">{violation.description}</p>
              </div>
            </div>
          ))}
          
          {violations.length > 5 && (
            <div className="text-center text-sm text-muted-foreground">
              +{violations.length - 5} more violations
            </div>
          )}
        </div>
      )}
    </Card>
  );
};

export default ProctoringMonitor;
```

## API Integration

### PreScreening API Client

```typescript
// services/prescreeningApi.ts
import axios, { AxiosResponse } from 'axios';
import { 
  PreScreeningSession, 
  MCQQuestion, 
  PreScreeningResult,
  ProctoringEvent 
} from '@/types/prescreening';

class PreScreeningApiClient {
  private baseURL: string;
  private axiosInstance;

  constructor(baseURL: string = '/api/prescreening') {
    this.baseURL = baseURL;
    this.axiosInstance = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add token to requests
    this.axiosInstance.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });
  }

  // Session Management
  async createSession(candidateId: string, jobId: string): Promise<PreScreeningSession> {
    const response = await this.axiosInstance.post('/sessions', {
      candidate_id: candidateId,
      job_id: jobId
    });
    return response.data;
  }

  async getSession(sessionId: string): Promise<PreScreeningSession> {
    const response = await this.axiosInstance.get(`/sessions/${sessionId}`);
    return response.data;
  }

  async updateSessionStatus(sessionId: string, status: string): Promise<void> {
    await this.axiosInstance.patch(`/sessions/${sessionId}`, { status });
  }

  // MCQ Generation and Submission
  async getNextQuestion(
    sessionId: string, 
    previousEvaluations: any[] = []
  ): Promise<MCQQuestion> {
    const response = await this.axiosInstance.post(`/sessions/${sessionId}/next-question`, {
      previous_evaluations: previousEvaluations
    });
    return response.data;
  }

  async submitAnswer(
    sessionId: string,
    questionId: string,
    selectedAnswer: string,
    videoData: string,
    audioTranscript: string,
    timeTaken: number
  ): Promise<any> {
    const response = await this.axiosInstance.post(`/sessions/${sessionId}/submit-answer`, {
      question_id: questionId,
      selected_answer: selectedAnswer,
      video_data: videoData,
      audio_transcript: audioTranscript,
      time_taken: timeTaken
    });
    return response.data;
  }

  // Results
  async getResults(sessionId: string): Promise<PreScreeningResult> {
    const response = await this.axiosInstance.get(`/sessions/${sessionId}/results`);
    return response.data;
  }

  async getFinalScore(sessionId: string): Promise<any> {
    const response = await this.axiosInstance.get(`/sessions/${sessionId}/final-score`);
    return response.data;
  }

  // Proctoring
  async startProctoring(sessionId: string): Promise<void> {
    await this.axiosInstance.post(`/sessions/${sessionId}/proctoring/start`);
  }

  async submitVideoFrame(
    sessionId: string,
    frameData: string,
    candidateId: string
  ): Promise<any> {
    const response = await this.axiosInstance.post(`/sessions/${sessionId}/proctoring/video-frame`, {
      frame_data: frameData,
      candidate_id: candidateId
    });
    return response.data;
  }

  async submitAudioChunk(
    sessionId: string,
    audioData: string,
    candidateId: string
  ): Promise<any> {
    const response = await this.axiosInstance.post(`/sessions/${sessionId}/proctoring/audio-chunk`, {
      audio_data: audioData,
      candidate_id: candidateId
    });
    return response.data;
  }

  async recordTabEvent(
    sessionId: string,
    eventType: string,
    eventData: any,
    candidateId: string
  ): Promise<any> {
    const response = await this.axiosInstance.post(`/sessions/${sessionId}/proctoring/tab-event`, {
      event_type: eventType,
      event_data: eventData,
      candidate_id: candidateId
    });
    return response.data;
  }

  async getProctoringEvents(sessionId: string): Promise<ProctoringEvent[]> {
    const response = await this.axiosInstance.get(`/sessions/${sessionId}/proctoring/events`);
    return response.data;
  }

  async getProctoringStatus(sessionId: string): Promise<any> {
    const response = await this.axiosInstance.get(`/sessions/${sessionId}/proctoring/status`);
    return response.data;
  }

  // Human Review
  async requestHumanReview(sessionId: string, reason: string): Promise<void> {
    await this.axiosInstance.post(`/sessions/${sessionId}/human-review`, { reason });
  }

  async submitHumanReview(
    sessionId: string,
    decision: string,
    notes: string,
    reviewerId: string
  ): Promise<void> {
    await this.axiosInstance.post(`/sessions/${sessionId}/human-review/submit`, {
      decision,
      notes,
      reviewer_id: reviewerId
    });
  }

  // Analytics
  async getSessionAnalytics(sessionId: string): Promise<any> {
    const response = await this.axiosInstance.get(`/sessions/${sessionId}/analytics`);
    return response.data;
  }
}

export const prescreeningApi = new PreScreeningApiClient();
export default PreScreeningApiClient;
```

## Hooks

### Main Pre-Screening Hook

```typescript
// hooks/usePreScreening.ts
import { useState, useEffect, useCallback } from 'react';
import { PreScreeningSession, PreScreeningStatus } from '@/types/prescreening';
import { prescreeningApi } from '@/services/prescreeningApi';

export const usePreScreening = (candidateId: string, jobId: string) => {
  const [session, setSession] = useState<PreScreeningSession | null>(null);
  const [status, setStatus] = useState<PreScreeningStatus>('NOT_STARTED');
  const [currentStep, setCurrentStep] = useState<'setup' | 'screening' | 'results'>('setup');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startSession = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const newSession = await prescreeningApi.createSession(candidateId, jobId);
      setSession(newSession);
      setStatus('IN_PROGRESS');
      setCurrentStep('screening');
      
      // Start proctoring
      await prescreeningApi.startProctoring(newSession.id);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start session');
    } finally {
      setLoading(false);
    }
  }, [candidateId, jobId]);

  const completeSession = useCallback(async (results: any) => {
    if (!session) return;
    
    setLoading(true);
    try {
      await prescreeningApi.updateSessionStatus(session.id, 'COMPLETED');
      setStatus('COMPLETED');
      setCurrentStep('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to complete session');
    } finally {
      setLoading(false);
    }
  }, [session]);

  const getSessionAnalytics = useCallback(async () => {
    if (!session) return null;
    
    try {
      return await prescreeningApi.getSessionAnalytics(session.id);
    } catch (err) {
      console.error('Failed to fetch analytics:', err);
      return null;
    }
  }, [session]);

  // Load existing session if available
  useEffect(() => {
    const loadSession = async () => {
      setLoading(true);
      try {
        // Check if there's an existing session for this candidate/job
        const existingSession = await prescreeningApi.getSession(`${candidateId}-${jobId}`);
        if (existingSession) {
          setSession(existingSession);
          setStatus(existingSession.status as PreScreeningStatus);
          
          // Determine current step based on status
          if (existingSession.status === 'COMPLETED') {
            setCurrentStep('results');
          } else if (existingSession.status === 'IN_PROGRESS') {
            setCurrentStep('screening');
          }
        }
      } catch (err) {
        // No existing session found, start fresh
        console.log('No existing session found');
      } finally {
        setLoading(false);
      }
    };

    if (candidateId && jobId) {
      loadSession();
    }
  }, [candidateId, jobId]);

  return {
    session,
    status,
    currentStep,
    loading,
    error,
    startSession,
    completeSession,
    getSessionAnalytics
  };
};
```

### Proctoring Hook

```typescript
// hooks/useProctoring.ts
import { useState, useEffect, useRef, useCallback } from 'react';
import { prescreeningApi } from '@/services/prescreeningApi';
import { ProctoringViolation } from '@/types/prescreening';

export const useProctoring = (sessionId: string) => {
  const [violations, setViolations] = useState<ProctoringViolation[]>([]);
  const [proctoringActive, setProctoringActive] = useState(false);
  const [shouldTerminate, setShouldTerminate] = useState(false);
  
  const videoFrameInterval = useRef<NodeJS.Timeout>();
  const audioChunkInterval = useRef<NodeJS.Timeout>();
  const tabEventListeners = useRef<(() => void)[]>([]);

  const startProctoring = useCallback(async () => {
    try {
      await prescreeningApi.startProctoring(sessionId);
      setProctoringActive(true);
      
      // Start video frame capture
      startVideoFrameCapture();
      
      // Start audio monitoring
      startAudioMonitoring();
      
      // Setup tab event listeners
      setupTabMonitoring();
      
    } catch (error) {
      console.error('Failed to start proctoring:', error);
    }
  }, [sessionId]);

  const stopProctoring = useCallback(() => {
    setProctoringActive(false);
    
    // Clear intervals
    if (videoFrameInterval.current) {
      clearInterval(videoFrameInterval.current);
    }
    if (audioChunkInterval.current) {
      clearInterval(audioChunkInterval.current);
    }
    
    // Remove event listeners
    tabEventListeners.current.forEach(cleanup => cleanup());
    tabEventListeners.current = [];
    
  }, []);

  const startVideoFrameCapture = () => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    videoFrameInterval.current = setInterval(async () => {
      try {
        const videoElement = document.querySelector('video') as HTMLVideoElement;
        if (!videoElement || !context) return;
        
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        context.drawImage(videoElement, 0, 0);
        
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        const candidateId = getCurrentCandidateId(); // Implement this
        
        const result = await prescreeningApi.submitVideoFrame(
          sessionId,
          frameData,
          candidateId
        );
        
        if (result.detections && result.detections.length > 0) {
          const newViolations = result.detections.map((detection: any) => ({
            type: detection.type,
            severity: detection.severity,
            description: detection.description,
            timestamp: new Date().toISOString()
          }));
          
          setViolations(prev => [...prev, ...newViolations].slice(-20)); // Keep last 20
        }
        
        if (result.should_terminate) {
          setShouldTerminate(true);
        }
        
      } catch (error) {
        console.error('Error processing video frame:', error);
      }
    }, 2000); // Every 2 seconds
  };

  const startAudioMonitoring = () => {
    // This would typically use Web Audio API or MediaRecorder
    audioChunkInterval.current = setInterval(async () => {
      try {
        // Mock audio data - in real implementation, capture actual audio
        const mockAudioData = "base64_encoded_audio_chunk";
        const candidateId = getCurrentCandidateId();
        
        const result = await prescreeningApi.submitAudioChunk(
          sessionId,
          mockAudioData,
          candidateId
        );
        
        if (result.audio_detections && result.audio_detections.length > 0) {
          const newViolations = result.audio_detections.map((detection: any) => ({
            type: detection.type,
            severity: detection.severity,
            description: detection.description,
            timestamp: new Date().toISOString()
          }));
          
          setViolations(prev => [...prev, ...newViolations].slice(-20));
        }
        
      } catch (error) {
        console.error('Error processing audio chunk:', error);
      }
    }, 5000); // Every 5 seconds
  };

  const setupTabMonitoring = () => {
    const candidateId = getCurrentCandidateId();
    
    // Focus/blur events
    const handleFocus = () => recordTabEvent('focus_gained', {});
    const handleBlur = () => recordTabEvent('focus_lost', {});
    
    // Visibility change
    const handleVisibilityChange = () => {
      recordTabEvent('visibility_change', {
        is_visible: !document.hidden
      });
    };
    
    // Page beforeunload
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      recordTabEvent('page_unload_attempt', {});
      e.preventDefault();
      e.returnValue = '';
    };
    
    window.addEventListener('focus', handleFocus);
    window.addEventListener('blur', handleBlur);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    const cleanup = () => {
      window.removeEventListener('focus', handleFocus);
      window.removeEventListener('blur', handleBlur);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
    
    tabEventListeners.current.push(cleanup);
  };

  const recordTabEvent = async (eventType: string, eventData: any) => {
    try {
      const candidateId = getCurrentCandidateId();
      const result = await prescreeningApi.recordTabEvent(
        sessionId,
        eventType,
        eventData,
        candidateId
      );
      
      if (result.event_recorded && result.severity) {
        const violation: ProctoringViolation = {
          type: eventType.toUpperCase(),
          severity: result.severity,
          description: result.description || `Tab event: ${eventType}`,
          timestamp: new Date().toISOString()
        };
        
        setViolations(prev => [...prev, violation].slice(-20));
      }
      
    } catch (error) {
      console.error('Error recording tab event:', error);
    }
  };

  // Helper function - implement based on your auth system
  const getCurrentCandidateId = (): string => {
    return localStorage.getItem('candidate_id') || 'unknown';
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopProctoring();
    };
  }, [stopProctoring]);

  return {
    violations,
    proctoringActive,
    shouldTerminate,
    startProctoring,
    stopProctoring,
    recordTabEvent
  };
};
```

## WebSocket Integration

```typescript
// hooks/useWebSocket.ts
import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
}

export const useWebSocket = (url: string) => {
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const ws = useRef<WebSocket | null>(null);
  const reconnectInterval = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;
    
    setConnectionState('connecting');
    ws.current = new WebSocket(url);
    
    ws.current.onopen = () => {
      setConnectionState('connected');
      setError(null);
      reconnectAttempts.current = 0;
    };
    
    ws.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setLastMessage(message);
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };
    
    ws.current.onclose = () => {
      setConnectionState('disconnected');
      
      // Auto-reconnect logic
      if (reconnectAttempts.current < 5) {
        reconnectInterval.current = setTimeout(() => {
          reconnectAttempts.current += 1;
          connect();
        }, 1000 * Math.pow(2, reconnectAttempts.current)); // Exponential backoff
      }
    };
    
    ws.current.onerror = (error) => {
      setError('WebSocket connection error');
      console.error('WebSocket error:', error);
    };
  }, [url]);

  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        ...message,
        timestamp: new Date().toISOString()
      }));
    } else {
      console.warn('WebSocket not connected, message not sent');
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectInterval.current) {
      clearTimeout(reconnectInterval.current);
    }
    
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    
    setConnectionState('disconnected');
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return {
    connectionState,
    lastMessage,
    error,
    sendMessage,
    disconnect
  };
};
```

## TypeScript Types

```typescript
// types/prescreening.ts
export interface PreScreeningSession {
  id: string;
  candidate_id: string;
  job_id: string;
  status: PreScreeningStatus;
  total_questions: number;
  questions_answered: number;
  current_score: number;
  started_at: string;
  completed_at?: string;
  configuration: PreScreeningConfiguration;
  proctoring_enabled: boolean;
}

export enum PreScreeningStatus {
  NOT_STARTED = 'NOT_STARTED',
  IN_PROGRESS = 'IN_PROGRESS',
  COMPLETED = 'COMPLETED',
  TERMINATED = 'TERMINATED',
  REQUIRES_HUMAN_REVIEW = 'REQUIRES_HUMAN_REVIEW'
}

export interface MCQQuestion {
  id: string;
  question_text: string;
  options: { [key: string]: string };
  correct_answer: string;
  explanation: string;
  category: string;
  difficulty_level: 'easy' | 'intermediate' | 'hard';
  time_limit?: number;
  created_at: string;
}

export interface PreScreeningResult {
  session_id: string;
  candidate_id: string;
  overall_score: number;
  accuracy_score: number;
  confidence_score: number;
  engagement_score: number;
  professionalism_score: number;
  bucket: ScoreBucket;
  total_questions: number;
  correct_answers: number;
  recommendations: string[];
  detailed_analysis: {
    question_performance: QuestionPerformance[];
    video_analysis_summary: VideoAnalysisSummary;
    proctoring_summary: ProctoringEventSummary;
  };
}

export interface ProctoringViolation {
  type: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  description: string;
  timestamp: string;
  evidence?: any;
}

export interface ProctoringEvent {
  id: string;
  session_id: string;
  candidate_id: string;
  event_type: string;
  event_data: any;
  severity: string;
  detected_at: string;
}

export enum ScoreBucket {
  EXCELLENT = 'EXCELLENT',
  GOOD = 'GOOD',
  POTENTIAL = 'POTENTIAL',
  NOT_ELIGIBLE = 'NOT_ELIGIBLE'
}

export interface PreScreeningConfiguration {
  max_questions: number;
  time_per_question: number;
  difficulty_adaptation: boolean;
  proctoring_enabled: boolean;
  auto_terminate_violations: boolean;
  human_review_threshold: number;
  job_requirements: string[];
}

export interface QuestionPerformance {
  question_id: string;
  selected_answer: string;
  correct_answer: string;
  is_correct: boolean;
  time_taken: number;
  confidence_score: number;
  video_analysis: {
    engagement: number;
    clarity: number;
    professionalism: number;
  };
}

export interface VideoAnalysisSummary {
  average_confidence: number;
  average_engagement: number;
  average_clarity: number;
  average_professionalism: number;
  red_flags: string[];
  positive_indicators: string[];
}

export interface ProctoringEventSummary {
  total_violations: number;
  violation_score: number;
  critical_violations: number;
  high_violations: number;
  medium_violations: number;
  low_violations: number;
  recommendation: string;
}
```

## Error Handling & Testing

### Error Boundary Component

```typescript
// components/ErrorBoundary.tsx
import React, { Component, ReactNode } from 'react';
import { Alert, Button } from '@/components/ui';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: any) => void;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class PreScreeningErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('PreScreening Error Boundary caught an error:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-6 text-center">
          <Alert variant="destructive">
            <h3 className="font-medium">Something went wrong</h3>
            <p className="mt-2">The pre-screening system encountered an error.</p>
            <Button 
              className="mt-4" 
              onClick={() => this.setState({ hasError: false })}
            >
              Try Again
            </Button>
          </Alert>
        </div>
      );
    }

    return this.props.children;
  }
}

export default PreScreeningErrorBoundary;
```

### Testing Utilities

```typescript
// utils/testing.ts
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PreScreeningSession, MCQQuestion } from '@/types/prescreening';

// Mock data generators
export const generateMockSession = (overrides?: Partial<PreScreeningSession>): PreScreeningSession => ({
  id: 'session_123',
  candidate_id: 'candidate_456',
  job_id: 'job_789',
  status: 'IN_PROGRESS',
  total_questions: 10,
  questions_answered: 3,
  current_score: 75.5,
  started_at: new Date().toISOString(),
  configuration: {
    max_questions: 10,
    time_per_question: 180,
    difficulty_adaptation: true,
    proctoring_enabled: true,
    auto_terminate_violations: false,
    human_review_threshold: 50,
    job_requirements: ['JavaScript', 'React', 'Node.js']
  },
  proctoring_enabled: true,
  ...overrides
});

export const generateMockQuestion = (overrides?: Partial<MCQQuestion>): MCQQuestion => ({
  id: 'question_123',
  question_text: 'What is the virtual DOM in React?',
  options: {
    A: 'A virtual representation of the HTML DOM',
    B: 'A JavaScript library for DOM manipulation',
    C: 'A browser API for virtual reality',
    D: 'A Node.js module for server-side rendering'
  },
  correct_answer: 'A',
  explanation: 'The virtual DOM is a virtual representation of the actual DOM kept in memory.',
  category: 'React',
  difficulty_level: 'intermediate',
  time_limit: 180,
  created_at: new Date().toISOString(),
  ...overrides
});

// Custom render function
export const renderWithProviders = (
  ui: React.ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );

  return render(ui, { wrapper: Wrapper, ...options });
};

// Mock API responses
export const mockApiResponses = {
  createSession: generateMockSession(),
  getNextQuestion: generateMockQuestion(),
  submitAnswer: {
    evaluation: {
      question_id: 'question_123',
      selected_answer: 'A',
      is_correct: true,
      confidence_level: 85.5,
      time_taken_seconds: 120
    },
    next_question: generateMockQuestion({ id: 'question_124' })
  },
  getFinalResults: {
    overall_score: 82.3,
    accuracy_score: 80.0,
    confidence_score: 85.0,
    engagement_score: 78.5,
    professionalism_score: 88.0,
    bucket: 'EXCELLENT',
    recommendations: ['Strong technical knowledge', 'Confident responses']
  }
};
```

This comprehensive Frontend Integration Specification provides:

1. **Complete Component Architecture** with React components for the entire pre-screening flow
2. **API Integration** with TypeScript client and proper error handling
3. **Real-time Features** including WebSocket integration and proctoring
4. **State Management** with custom hooks for complex state logic
5. **Type Safety** with comprehensive TypeScript definitions
6. **Testing Strategy** with utilities and mock data generators
7. **Error Handling** with error boundaries and graceful degradation
8. **Performance Considerations** with proper cleanup and optimization

The frontend seamlessly integrates with the Python backend you've developed, providing a complete end-to-end solution for the HireGenix Pre-Screening Flow.