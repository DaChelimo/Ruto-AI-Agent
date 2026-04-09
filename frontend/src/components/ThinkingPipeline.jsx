import styles from './ThinkingPipeline.module.css';

// Thin blinking cursor shown on the active step label
function Cursor() {
  return <span className={styles.cursor} aria-hidden="true">_</span>;
}

// A single pipeline step row
function PipelineStep({ step, isLast }) {
  const { label, icon, status } = step;
  const isActive    = status === 'active';
  const isCompleted = status === 'completed';
  const isPending   = status === 'pending';

  return (
    <li
      className={[
        styles.step,
        isActive    ? styles.stepActive    : '',
        isCompleted ? styles.stepCompleted : '',
        isPending   ? styles.stepPending   : '',
      ].filter(Boolean).join(' ')}
      aria-label={`${label}: ${status}`}
    >
      {/* Vertical connecting line — hidden on last item */}
      {!isLast && <span className={styles.connector} aria-hidden="true" />}

      {/* Status dot */}
      <span className={styles.dot} aria-hidden="true">
        {isCompleted ? '●' : isActive ? '●' : '○'}
      </span>

      {/* Icon */}
      <span className={styles.icon} aria-hidden="true">{icon}</span>

      {/* Label */}
      <span className={styles.label}>
        {label}
        {isActive && <Cursor />}
        {!isPending && <span className={styles.chevron} aria-hidden="true"> ∨</span>}
      </span>
    </li>
  );
}

export default function ThinkingPipeline({ steps, visible }) {
  return (
    <aside
      className={[styles.panel, visible ? styles.visible : ''].join(' ')}
      aria-label="Processing pipeline"
      aria-live="polite"
    >
      <div className={styles.inner}>
        <p className={styles.header}>TASK INITIATED</p>
        <ul className={styles.list}>
          {steps.map((step, i) => (
            <PipelineStep
              key={step.label}
              step={step}
              isLast={i === steps.length - 1}
            />
          ))}
        </ul>
      </div>
    </aside>
  );
}
