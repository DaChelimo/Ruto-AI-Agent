import { useChat } from './hooks/useChat';
import Header from './components/Header';
import ThinkingPipeline from './components/ThinkingPipeline';
import ChatArea from './components/ChatArea';
import InputBar from './components/InputBar';
import styles from './App.module.css';

// ── Toast notification ────────────────────────────────────────────────────────
function Toast({ message }) {
  return (
    <div className={styles.toast} role="status" aria-live="polite">
      {message}
    </div>
  );
}

export default function App() {
  const {
    messages,
    pipelineSteps,
    isProcessing,
    isEnded,
    toast,
    sendMessage,
    startNewChat,
    retryLastMessage,
  } = useChat();

  // Pipeline panel is visible once there are steps to show
  const showPipeline = pipelineSteps.length > 0;

  return (
    <div className={styles.app}>
      <Header onNewChat={startNewChat} />

      {toast && <Toast message={toast} />}

      <div className={styles.main}>
        {/* Left panel: thinking pipeline */}
        <ThinkingPipeline
          steps={pipelineSteps}
          visible={showPipeline}
        />

        {/* Right panel: conversation */}
        <ChatArea
          messages={messages}
          pipelineSteps={pipelineSteps}
          isProcessing={isProcessing}
          isEnded={isEnded}
          onChipClick={sendMessage}
          onNewChat={startNewChat}
          onRetry={retryLastMessage}
        />
      </div>

      <InputBar
        onSend={sendMessage}
        disabled={isProcessing || isEnded}
        isEnded={isEnded}
      />
    </div>
  );
}
