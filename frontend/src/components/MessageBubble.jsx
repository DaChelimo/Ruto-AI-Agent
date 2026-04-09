import rutoIcon from '../assets/ruto-icon.svg';
import styles from './MessageBubble.module.css';

const SUGGESTED_PROMPTS = [
  'Tell me about your rise to the presidency',
  'What is your education background?',
  'What are you working on currently?',
  'What do you think about corruption?',
];

// ── User bubble ───────────────────────────────────────────────────────────────
function UserBubble({ text }) {
  return (
    <div className={styles.userRow}>
      <div className={styles.userBubble}>
        <p className={styles.text}>{text}</p>
      </div>
    </div>
  );
}

// ── Agent bubble ──────────────────────────────────────────────────────────────
function AgentBubble({ text }) {
  return (
    <div className={styles.agentRow}>
      <img
        src={rutoIcon}
        alt="Ruto Agent"
        className={styles.avatar}
        width={28}
        height={28}
      />
      <div className={styles.agentBubble}>
        <p className={styles.text}>{text}</p>
      </div>
    </div>
  );
}

// ── Error bubble ──────────────────────────────────────────────────────────────
function ErrorBubble({ text, onRetry }) {
  return (
    <div className={styles.errorRow}>
      <div className={styles.errorBubble}>
        <p className={styles.errorText}>{text}</p>
        {onRetry && (
          <button className={styles.retryBtn} onClick={onRetry}>
            Retry
          </button>
        )}
      </div>
    </div>
  );
}

// ── Welcome screen ────────────────────────────────────────────────────────────
export function WelcomeScreen({ onChipClick }) {
  return (
    <div className={styles.welcome}>
      <img
        src={rutoIcon}
        alt="William Ruto"
        className={styles.welcomeAvatar}
        width={48}
        height={48}
      />
      <p className={styles.welcomeName}>WILLIAM RUTO</p>
      <p className={styles.welcomeSub}>President of Kenya</p>
      <div className={styles.chipRow}>
        {SUGGESTED_PROMPTS.map(prompt => (
          <button
            key={prompt}
            className={styles.chip}
            onClick={() => onChipClick(prompt)}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── End state ─────────────────────────────────────────────────────────────────
export function ConversationEndedLabel({ onNewChat }) {
  return (
    <div className={styles.endedContainer}>
      <span className={styles.endedLabel}>CONVERSATION ENDED</span>
      <button className={styles.newConvBtn} onClick={onNewChat}>
        Start New Conversation
      </button>
    </div>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────
export default function MessageBubble({ message, onRetry }) {
  if (message.role === 'user')  return <UserBubble  text={message.text} />;
  if (message.role === 'agent') return <AgentBubble text={message.text} />;
  if (message.role === 'error') return <ErrorBubble text={message.text} onRetry={onRetry} />;
  return null;
}
