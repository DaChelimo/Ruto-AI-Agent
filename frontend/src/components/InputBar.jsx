import { useState, useRef, useCallback } from 'react';
import styles from './InputBar.module.css';

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M8 13V3M8 3L3.5 7.5M8 3L12.5 7.5" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

export default function InputBar({ onSend, disabled, isEnded }) {
  const [value, setValue] = useState('');
  const textareaRef = useRef(null);

  const canSend = value.trim().length > 0 && !disabled;

  // Auto-resize textarea up to ~5 lines
  const resize = useCallback((el) => {
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 128) + 'px';
  }, []);

  const handleChange = (e) => {
    setValue(e.target.value);
    resize(e.target);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const submit = () => {
    if (!canSend) return;
    onSend(value);
    setValue('');
    // Reset textarea height and return focus so the next message can be typed immediately
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.focus();
    }
  };

  return (
    <div className={styles.bar}>
      <div className={styles.inner}>
        <textarea
          ref={textareaRef}
          className={styles.input}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={isEnded ? 'Conversation ended' : 'Ask or follow up…'}
          disabled={disabled}
          rows={1}
          aria-label="Message input"
        />
        <button
          className={styles.sendBtn}
          onClick={submit}
          disabled={!canSend}
          aria-label="Send message"
        >
          <SendIcon />
        </button>
      </div>
    </div>
  );
}
