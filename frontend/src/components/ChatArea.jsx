import { useRef, useEffect } from 'react';
import MessageBubble, {
  WelcomeScreen,
  ConversationEndedLabel,
} from './MessageBubble';
import styles from './ChatArea.module.css';

// Compact mobile pipeline strip — shows dots above agent response
function MobilePipelineStrip({ steps }) {
  if (!steps || steps.length === 0) return null;
  return (
    <div className={styles.mobileStrip} aria-label="Processing steps" aria-live="polite">
      {steps.map((step, i) => (
        <span
          key={step.label}
          className={[
            styles.mobileDot,
            step.status === 'completed' ? styles.mobileDotDone    : '',
            step.status === 'active'    ? styles.mobileDotActive  : '',
            step.status === 'pending'   ? styles.mobileDotPending : '',
          ].filter(Boolean).join(' ')}
          title={step.label}
          aria-label={step.label}
        >
          ●
        </span>
      ))}
      <span className={styles.mobileStepLabel}>
        {steps.find(s => s.status === 'active')?.label ??
         (steps.every(s => s.status === 'completed') ? 'DONE' : '')}
      </span>
    </div>
  );
}

// Typing indicator shown while the agent is processing (no pipeline yet / mobile)
function TypingIndicator() {
  return (
    <div className={styles.typingRow}>
      <div className={styles.typingBubble}>
        <span className={styles.dot1} />
        <span className={styles.dot2} />
        <span className={styles.dot3} />
      </div>
    </div>
  );
}

export default function ChatArea({
  messages,
  pipelineSteps,
  isProcessing,
  isEnded,
  onChipClick,
  onNewChat,
  onRetry,
}) {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom whenever messages or processing state changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isProcessing]);

  const isEmpty = messages.length === 0 && !isProcessing;

  return (
    <main className={styles.area}>
      {isEmpty ? (
        <WelcomeScreen onChipClick={onChipClick} />
      ) : (
        <div className={styles.messages}>
          {messages.map((msg, i) => {
            const isLastAgent = msg.role === 'agent' && i === messages.length - 1;
            return (
              <div key={msg.id}>
                {/* Mobile-only: pipeline strip above the last agent response */}
                {isLastAgent && pipelineSteps.length > 0 && (
                  <MobilePipelineStrip steps={pipelineSteps} />
                )}
                <MessageBubble
                  message={msg}
                  onRetry={msg.role === 'error' ? onRetry : undefined}
                />
              </div>
            );
          })}

          {/* Typing indicator while processing */}
          {isProcessing && <TypingIndicator />}

          {/* Conversation ended state */}
          {isEnded && <ConversationEndedLabel onNewChat={onNewChat} />}

          {/* Scroll sentinel */}
          <div ref={bottomRef} />
        </div>
      )}
    </main>
  );
}
