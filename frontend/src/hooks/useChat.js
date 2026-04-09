import { useState, useRef, useEffect, useCallback } from 'react';
import { startSession, sendChat } from '../config';

// Full pipeline shown during simulation (before API responds)
const FULL_PIPELINE = [
  { label: 'CLASSIFYING MESSAGE',   icon: '⊡' },
  { label: 'SEARCHING MEMORY',      icon: '⊙' },
  { label: 'CHECKING SUFFICIENCY',  icon: '⊘' },
  { label: 'WEB SEARCH',            icon: '⊕' },
  { label: 'BUILDING CONTENT PLAN', icon: '⊞' },
  { label: 'STYLING RESPONSE',      icon: '⊟' },
];

// Cumulative delays (ms) for each step activation during simulation
const STEP_DELAYS = [0, 800, 2000, 3500, 4500, 5300];

// Icon map for steps returned by the backend
const STEP_ICON_MAP = {
  'CLASSIFYING MESSAGE':   '⊡',
  'SEARCHING MEMORY':      '⊙',
  'CHECKING SUFFICIENCY':  '⊘',
  'WEB SEARCH':            '⊕',
  'BUILDING CONTENT PLAN': '⊞',
  'STYLING RESPONSE':      '⊟',
  'ENDING CONVERSATION':   '⊠',
};

function getStepIcon(label) {
  return STEP_ICON_MAP[label] ?? '○';
}

let msgCounter = 0;
function uid(prefix) {
  return `${prefix}-${++msgCounter}-${Date.now()}`;
}

export function useChat() {
  const [messages, setMessages]         = useState([]);
  const [pipelineSteps, setPipelineSteps] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isEnded, setIsEnded]           = useState(false);
  const [toast, setToast]               = useState(null);

  // Refs for values needed inside async callbacks without stale closure issues
  const sessionIdRef      = useRef(null);
  const timeoutRefs       = useRef([]);
  const lastUserMsgRef    = useRef(null);
  const toastTimerRef     = useRef(null);

  // ── Toast ─────────────────────────────────────────────────────────────────
  const showToast = useCallback((msg) => {
    clearTimeout(toastTimerRef.current);
    setToast(msg);
    toastTimerRef.current = setTimeout(() => setToast(null), 4500);
  }, []);

  // ── Animation helpers ─────────────────────────────────────────────────────
  const clearAnimationTimeouts = useCallback(() => {
    timeoutRefs.current.forEach(clearTimeout);
    timeoutRefs.current = [];
  }, []);

  const startPipelineAnimation = useCallback(() => {
    clearAnimationTimeouts();

    // All steps start as pending
    setPipelineSteps(FULL_PIPELINE.map(s => ({ ...s, status: 'pending' })));

    // Activate each step at its cumulative delay
    STEP_DELAYS.forEach((delay, i) => {
      const id = setTimeout(() => {
        setPipelineSteps(prev =>
          prev.map((step, idx) => {
            if (idx < i)  return { ...step, status: 'completed' };
            if (idx === i) return { ...step, status: 'active' };
            return step;
          })
        );
      }, delay);
      timeoutRefs.current.push(id);
    });
  }, [clearAnimationTimeouts]);

  // Resolve animation with real steps from the backend (or snap all if absent)
  const resolvePipeline = useCallback((actualStepLabels) => {
    clearAnimationTimeouts();

    if (actualStepLabels && actualStepLabels.length > 0) {
      setPipelineSteps(
        actualStepLabels.map(label => ({
          label,
          icon: getStepIcon(label),
          status: 'completed',
        }))
      );
    } else {
      // Fallback: snap all existing steps to completed
      setPipelineSteps(prev => prev.map(s => ({ ...s, status: 'completed' })));
    }
  }, [clearAnimationTimeouts]);

  // ── Session management ────────────────────────────────────────────────────
  const initSession = useCallback(async () => {
    try {
      const data = await startSession();
      sessionIdRef.current = data.session_id;
      return data.session_id;
    } catch {
      return null;
    }
  }, []);

  const startNewChat = useCallback(async () => {
    clearAnimationTimeouts();
    await initSession();
    setMessages([]);
    setPipelineSteps([]);
    setIsEnded(false);
    lastUserMsgRef.current = null;
  }, [clearAnimationTimeouts, initSession]);

  // Auto-start session when the hook first mounts
  useEffect(() => {
    initSession();
    return () => {
      clearAnimationTimeouts();
      clearTimeout(toastTimerRef.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Send message ──────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (text) => {
    const trimmed = (text || '').trim();
    if (!trimmed || isProcessing || isEnded) return;

    lastUserMsgRef.current = trimmed;

    // Add user bubble immediately
    setMessages(prev => [
      ...prev,
      { id: uid('user'), role: 'user', text: trimmed, timestamp: new Date() },
    ]);
    setIsProcessing(true);
    startPipelineAnimation();

    try {
      // Lazily ensure we have a session
      let sid = sessionIdRef.current;
      if (!sid) {
        sid = await initSession();
        if (!sid) throw new Error('Could not start session. Is the backend running?');
      }

      const data = await sendChat(sid, trimmed);

      // Settle the pipeline with real steps
      resolvePipeline(data.pipeline_steps);

      if (!data.response) throw new Error('empty_response');

      setMessages(prev => [
        ...prev,
        { id: uid('agent'), role: 'agent', text: data.response, timestamp: new Date() },
      ]);

      if (data.ended) {
        setIsEnded(true);
      }
    } catch (err) {
      clearAnimationTimeouts();
      // Fade out any active animation steps back to pending
      setPipelineSteps(prev =>
        prev.map(s => s.status === 'active' ? { ...s, status: 'pending' } : s)
      );

      const msg = err.message ?? '';

      if (msg.includes('404')) {
        showToast('Session expired — started a new conversation');
        await startNewChat();
      } else {
        const errorText =
          msg === 'empty_response'
            ? 'The agent didn\'t respond. Please try again.'
            : 'Something went wrong. Please try again.';

        setMessages(prev => [
          ...prev,
          { id: uid('error'), role: 'error', text: errorText, timestamp: new Date() },
        ]);
      }
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing, isEnded, startPipelineAnimation, resolvePipeline, clearAnimationTimeouts, initSession, showToast, startNewChat]);

  // ── Retry ─────────────────────────────────────────────────────────────────
  const retryLastMessage = useCallback(() => {
    if (!lastUserMsgRef.current || isProcessing) return;
    // Remove the trailing error bubble, then re-send
    setMessages(prev => {
      const last = prev[prev.length - 1];
      return last?.role === 'error' ? prev.slice(0, -1) : prev;
    });
    sendMessage(lastUserMsgRef.current);
  }, [isProcessing, sendMessage]);

  return {
    messages,
    pipelineSteps,
    isProcessing,
    isEnded,
    toast,
    sendMessage,
    startNewChat,
    retryLastMessage,
  };
}
