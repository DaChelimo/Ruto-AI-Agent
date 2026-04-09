import styles from './Header.module.css';

function LogoIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <polygon
        points="10,1.5 12.3,7.7 19,10 12.3,12.3 10,18.5 7.7,12.3 1,10 7.7,7.7"
        stroke="#888888"
        strokeWidth="1.4"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

export default function Header({ onNewChat }) {
  return (
    <header className={styles.header}>
      <div className={styles.left}>
        <LogoIcon />
        <span className={styles.title}>RUTO AGENT</span>
      </div>
      <div className={styles.right}>
        <button
          className={styles.newChatBtn}
          onClick={onNewChat}
          aria-label="Start a new conversation"
        >
          New Chat
        </button>
      </div>
    </header>
  );
}
