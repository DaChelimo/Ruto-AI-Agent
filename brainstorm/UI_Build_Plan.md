# Ruto Agent — Frontend Build Plan

This document is a step-by-step implementation guide for building the web interface for the William Ruto conversational agent. It connects to an existing FastAPI backend deployed on Render.

---

## 0. Reference Image Analysis

Three reference images define the visual language. Here is what was extracted from each:

### layout_structure.png — Full Page Layout
- **Header bar**: logo icon + "CHAT" label on the left; action buttons on the right
- **Two-column main area**:
  - LEFT: a vertical "thinking pipeline" showing processing steps (each with an icon, an ALL-CAPS label, a dropdown chevron, and status indicators). Steps are connected by a thin vertical timeline line. A final output paragraph appears at the bottom of the pipeline.
  - RIGHT: the user's prompt text, displayed plainly
- **Input bar** at the bottom: placeholder text "Ask or follow up...", a (+) attachment button on the left, a send button on the right
- **Colors**: pure white background (#FFFFFF), no colored accents except status indicators. Very minimal, high white-space

### classy_font_text.png — Thinking/Pipeline Font
- **Font**: monospace, ALL UPPERCASE, with generous letter-spacing (~0.08em)
- **Closest match**: **Space Mono** (Google Fonts) — geometric monospace with clean, distinctive letterforms. The "W", "A", "R" shapes and overall regularity are characteristic of Space Mono.
- **Fallback stack**: `'Space Mono', 'IBM Plex Mono', 'Roboto Mono', monospace`
- **Color**: medium gray (#888888)
- **Weight**: 400 (regular)
- **Usage**: pipeline step labels, status text, "TASK INITIATED" header
- **Status indicators**: filled circle (●) = completed/active, empty circle (○) = pending/subtask
- **Dropdown chevrons**: small "∨" after expandable items
- **Connecting line**: thin vertical gray line running along the left side of the pipeline

### regular_text.png — Message/Content Font
- **Font**: sans-serif, sentence case, rounded and humanist
- **Closest match**: **DM Sans** (Google Fonts) — rounded, friendly, wide letter proportions with single-story "a" and generous spacing
- **Fallback stack**: `'DM Sans', 'Plus Jakarta Sans', 'Inter', sans-serif`
- **Color**: dark charcoal (#333333) for primary text
- **Weight**: 400 for body text, 500 for emphasis
- **Card style**: white background, border-radius ~14px, very subtle box-shadow (`0 1px 3px rgba(0,0,0,0.08)`)
- **Icon**: small rounded-square icon (white bg, colored glyph) in the top-left of the message card

---

## 1. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Framework | React 18 + Vite | Fast bundler, hot reload, standard for UI work |
| Styling | Plain CSS modules (one `.module.css` per component) | No extra deps, full control over the exact design tokens |
| Fonts | Google Fonts: Space Mono + DM Sans | Free, CDN-hosted, match the references precisely |
| HTTP client | Native `fetch()` | No need for Axios — the API has 2 endpoints |
| Deployment | Vercel (free tier) or same Render instance serving static files | Vercel is zero-config for Vite; Render can serve via FastAPI's `StaticFiles` |
| State | React `useState` + `useRef` | No Redux needed — the state is just a message list + session ID |

---

## 2. Project Scaffolding

Create the frontend as a separate directory inside the existing repo:

```
celebrity_agent/
├── src/                    # existing Python backend
├── data/                   # existing data
├── frontend/               # NEW — React app
│   ├── public/
│   │   └── favicon.svg
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── App.module.css
│   │   ├── config.js              # API base URL
│   │   ├── hooks/
│   │   │   └── useChat.js         # session + message state logic
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── Header.module.css
│   │   │   ├── ChatArea.jsx
│   │   │   ├── ChatArea.module.css
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── MessageBubble.module.css
│   │   │   ├── ThinkingPipeline.jsx
│   │   │   ├── ThinkingPipeline.module.css
│   │   │   ├── InputBar.jsx
│   │   │   └── InputBar.module.css
│   │   └── assets/
│   │       └── ruto-icon.svg      # agent avatar
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
└── requirements.txt        # existing
```

### Scaffolding commands:
```bash
cd celebrity_agent
npm create vite@latest frontend -- --template react
cd frontend
npm install
```

---

## 3. Design Tokens

Create a file `frontend/src/tokens.css` that is imported globally. Every component references these variables — never hardcode colors or fonts.

```css
:root {
  /* ── Fonts ─────────────────────────────────────────── */
  --font-mono: 'Space Mono', 'IBM Plex Mono', 'Roboto Mono', monospace;
  --font-sans: 'DM Sans', 'Plus Jakarta Sans', 'Inter', sans-serif;

  /* ── Colors ────────────────────────────────────────── */
  --color-bg:             #FFFFFF;
  --color-bg-subtle:      #F7F7F8;    /* very faint gray for alternating areas */
  --color-text-primary:   #1A1A1A;    /* near-black for agent responses + user text */
  /* --color-text-secondary: #888888;   /* medium gray for pipeline steps */
  --color-text-secondary: #777676;    /* medium gray for pipeline steps */
  --color-text-muted:     #BBBBBB;    /* placeholder text, timestamps */
  --color-border:         #E8E8E8;    /* subtle separators */
  --color-accent:         #2B2B2B;    /* dark accent for active states, send button */
  --color-pipeline-line:  #DDDDDD;    /* vertical connecting line in pipeline */
  --color-status-done:    #333333;    /* filled circle ● */
  --color-status-active:  #888888;    /* pulsing indicator */
  --color-status-pending: #CCCCCC;    /* empty circle ○ */
  --color-card-shadow:    rgba(0, 0, 0, 0.06);

  /* ── Spacing ───────────────────────────────────────── */
  --space-xs:  4px;
  --space-sm:  8px;
  --space-md:  16px;
  --space-lg:  24px;
  --space-xl:  40px;
  --space-2xl: 64px;

  /* ── Radii ─────────────────────────────────────────── */
  --radius-sm:  8px;
  --radius-md:  14px;
  --radius-lg:  20px;
  --radius-pill: 9999px;

  /* ── Sizing ────────────────────────────────────────── */
  --header-height:   56px;
  --input-bar-height: 64px;
  --max-content-width: 820px;
  --pipeline-width: 380px;
}
```

### Google Fonts — add to `index.html` `<head>`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
```

---

## 4. Global Reset — `frontend/src/index.css`

```css
@import './tokens.css';

*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body, #root {
  height: 100%;
  font-family: var(--font-sans);
  color: var(--color-text-primary);
  background: var(--color-bg);
  -webkit-font-smoothing: antialiased;
}
```

---

## 5. Component Specifications

### 5.1 `App.jsx` — Layout Shell

The root layout has three stacked sections: header, main content area, input bar. The main content area is a two-region layout:

```
┌─────────────────────────────────────────────────────────┐
│  Header (fixed top)                                     │
├─────────────────────┬───────────────────────────────────┤
│  ThinkingPipeline   │   ChatArea                        │
│  (left, 380px)      │   (right, fills remaining)        │
│  only visible when  │   scrollable message list         │
│  agent is thinking  │                                   │
│  or just responded  │                                   │
├─────────────────────┴───────────────────────────────────┤
│  InputBar (fixed bottom)                                │
└─────────────────────────────────────────────────────────┘
```

- The ThinkingPipeline panel slides in from the left when the agent is processing, and stays visible alongside the most recent agent response.
- On mobile (< 768px), the ThinkingPipeline collapses into an expandable section above the agent's response instead of a side panel.
- The ChatArea scrolls independently and auto-scrolls to the bottom on new messages.

### 5.2 `Header.jsx`

- Height: `var(--header-height)` (56px)
- Left side: an SVG icon/logo + the text "RUTO AGENT" in Space Mono, uppercase, letter-spacing 0.1em, color `var(--color-text-secondary)`
- Right side: a "New Chat" button (outlined, small, rounded) that calls `POST /session/start` and clears the message list
- Bottom border: 1px solid `var(--color-border)`

### 5.3 `ChatArea.jsx`

- Takes the full remaining height between Header and InputBar
- Contains a scrollable list of `MessageBubble` components
- On initial load (no messages yet): show a centered welcome state with the agent's name and a brief subtitle in Space Mono, plus 3-4 suggested prompt chips the user can click (e.g., "Tell me about your rise to presidency", "What is your education background?", "What are you working on currently?")
- Auto-scrolls to bottom when a new message is added (use a `useRef` on a sentinel div at the bottom + `scrollIntoView`)

### 5.4 `MessageBubble.jsx`

Two variants based on `role` prop ("user" or "agent"):

**User bubble:**
- Aligned to the RIGHT side
- Background: `var(--color-bg-subtle)` (#F7F7F8)
- Border-radius: `var(--radius-md)` (14px), with bottom-right corner set to `var(--space-xs)` (4px) for a chat-bubble tail effect
- Font: DM Sans, 400 weight, `var(--color-text-primary)`
- Max-width: 70% of the ChatArea width
- Padding: 16px 20px

**Agent bubble:**
- Aligned to the LEFT side
- Background: `var(--color-bg)` (white)
- Border: 1px solid `var(--color-border)`
- Border-radius: `var(--radius-md)` (14px), with bottom-left corner set to `var(--space-xs)` (4px)
- Small avatar icon in the top-left corner (a rounded square with the Kenyan flag colors or a neutral icon — 28x28px, border-radius 8px)
- Font: DM Sans, 400 weight, `var(--color-text-primary)`
- Max-width: 70% of the ChatArea width
- Padding: 16px 20px
- Subtle shadow: `0 1px 3px var(--color-card-shadow)`

### 5.5 `ThinkingPipeline.jsx`

This is the signature component — it shows the agent's processing steps in real time using the monospace "classy" font from the reference.

**Structure:**
- Width: `var(--pipeline-width)` (380px)
- Background: `var(--color-bg)`
- Left border: 1px solid `var(--color-border)`
- Padding: `var(--space-lg)` (24px)

**Header:** "TASK INITIATED" in Space Mono, uppercase, letter-spacing 0.08em, color `var(--color-text-secondary)`, font-size 11px

**Pipeline steps:** a vertical list of step items. Each step has:
- A status indicator on the left:
  - Completed: filled circle ● in `var(--color-status-done)`
  - Active/running: filled circle ● with a CSS pulse animation in `var(--color-status-active)`
  - Pending: empty circle ○ in `var(--color-status-pending)`
- An icon (small, 14px) — use simple Unicode or SVG icons:
  - CLASSIFYING → `⟐` or a tag icon
  - SEARCHING MEMORY → `◉` or a search icon
  - CHECKING SUFFICIENCY → `⟁` or a check icon
  - WEB SEARCH → `⊕` or a globe icon
  - BUILDING PLAN → `✎` or a pencil icon
  - STYLING RESPONSE → `◈` or a pen icon
- The step label in Space Mono, uppercase, letter-spacing 0.08em, font-size 12px, color `var(--color-text-secondary)`
- A dropdown chevron (∨) — for now this is decorative (not expandable), but the structure should support expansion later
- A thin vertical connecting line (2px wide, `var(--color-pipeline-line)`) running down the left side, connecting each step's indicator

**The six pipeline steps for the Ruto agent are, in order:**
1. CLASSIFYING MESSAGE
2. SEARCHING MEMORY (only for FACTUAL / HYBRID)
3. CHECKING SUFFICIENCY (only for FACTUAL / HYBRID)
4. WEB SEARCH (only if memory was insufficient)
5. BUILDING CONTENT PLAN
6. STYLING RESPONSE

Steps 2-4 are conditional — they only appear when the message is FACTUAL or HYBRID. For CONVERSATIONAL messages, the pipeline shows only steps 1, 5, 6.

**Animation behavior:**
- Steps appear one at a time with a staggered fade-in (200ms delay between each)
- The active step has a subtle pulse animation on its indicator dot
- When a step completes, its dot transitions from pulsing to solid with a quick scale animation
- The connecting line grows downward as each step appears (use CSS `height` transition)

### 5.6 `InputBar.jsx`

- Fixed to the bottom of the viewport
- Height: `var(--input-bar-height)` (64px)
- Background: `var(--color-bg)`
- Top border: 1px solid `var(--color-border)`
- Centered content with max-width matching the content area
- Contains:
  - A text input field: full-width, no visible border, font DM Sans 400, placeholder "Ask or follow up..." in `var(--color-text-muted)`
  - A send button on the right: a dark circle (`var(--color-accent)`) with a white arrow-up icon (↑), 36px diameter
  - The send button is disabled (opacity 0.3) when the input is empty or the agent is currently processing
- On Enter keypress: submit the message (same as clicking send)
- On Shift+Enter: newline (allow multiline input)

---

## 6. State Management — `useChat.js` Hook

This custom hook manages all conversation state and API communication:

```
useChat() returns:
  messages:       array of { id, role, text, timestamp }
  pipelineSteps:  array of { label, status } — drives ThinkingPipeline
  isProcessing:   boolean — true while waiting for agent response
  isEnded:        boolean — true if the conversation has been signed off
  sessionId:      string | null
  sendMessage(text):  function — sends user message, triggers pipeline animation, calls API
  startNewChat():     function — calls /session/start, resets all state
```

### `sendMessage(text)` flow:

1. Add a user message to `messages` with `role: "user"`
2. Set `isProcessing = true`
3. Start the pipeline animation (set steps to their initial states, begin stepping through them with timed delays — see Section 7)
4. Call `POST /chat` with `{ session_id, user_message: text }`
5. When the response arrives:
   a. Complete all remaining pipeline steps instantly (snap all to "completed")
   b. Add an agent message to `messages` with `role: "agent"` and the response text
   c. Set `isProcessing = false`
   d. If `ended === true`, set `isEnded = true`
6. On error: add a system message to `messages` with error text, set `isProcessing = false`

### `startNewChat()` flow:

1. Call `POST /session/start`
2. Save the returned `session_id`
3. Clear `messages` to empty array
4. Set `isEnded = false`
5. Reset `pipelineSteps` to empty

---

## 7. Pipeline Animation Timing

Since the backend does not stream intermediate steps (it returns a single final response), the ThinkingPipeline uses **timed simulation** — it steps through the pipeline labels at realistic intervals while the API call is in flight.

The timing should approximate real pipeline execution:
| Step | Delay after previous | Rationale |
|---|---|---|
| CLASSIFYING MESSAGE | 0ms (immediate) | First step, appears instantly |
| SEARCHING MEMORY | 800ms | Embedding + cosine similarity |
| CHECKING SUFFICIENCY | 1200ms | LLM call |
| WEB SEARCH | 1500ms | Network call to Tavily (only if needed) |
| BUILDING CONTENT PLAN | 1000ms | LLM planning call |
| STYLING RESPONSE | 800ms | LLM style call |

Implementation approach:
- When `sendMessage` is called, start a `setTimeout` chain that progressively sets each step's status from "pending" → "active" → "completed"
- If the API response arrives before the animation finishes, immediately snap all steps to "completed" (the real response takes priority over the animation)
- If the animation finishes before the API response, hold the last step as "active" (pulsing) until the response arrives

For CONVERSATIONAL messages (where memory retrieval and sufficiency checking are skipped), show only 3 steps:
1. CLASSIFYING MESSAGE
2. BUILDING CONTENT PLAN
3. STYLING RESPONSE

To determine which pipeline to show: always start with the full 6-step pipeline. Since the frontend does not know the classification result before the backend responds, show all steps. This is acceptable because the pipeline is a visual affordance, not a literal log.

ALTERNATIVE (more accurate): Modify the backend to return `message_type` in the response (see Section 9). Then, on the NEXT message, the frontend knows the pattern to expect and can adjust. For the first message, always show the full pipeline.

---

## 8. API Integration — `config.js`

```js
// frontend/src/config.js
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function startSession() {
  const res = await fetch(`${API_BASE_URL}/session/start`, { method: "POST" });
  if (!res.ok) throw new Error(`Start session failed: ${res.status}`);
  return res.json(); // { session_id, message }
}

export async function sendChat(sessionId, userMessage) {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, user_message: userMessage }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json(); // { response, ended, session_id }
}
```

### Environment variable for deployment:
- Local dev: no env var needed, defaults to `http://localhost:8000`
- Production (Vercel): set `VITE_API_URL` to `https://your-render-app.onrender.com`
- The backend already has CORS set to `allow_origins=["*"]`, so cross-origin requests work out of the box

---

## 9. Backend Modification — Return Pipeline Metadata

To make the ThinkingPipeline more accurate, add `pipeline_steps` metadata to the API response. This requires a small change to two files:

### `src/content_module.py` — track which steps ran:

Modify `content_step()` to return a `steps` list:

```python
def content_step(memory: MemoryStore, user_message: str) -> dict:
    steps = []

    message_type = classify_message(user_message)
    steps.append("CLASSIFYING MESSAGE")
    retrieved = []

    if message_type == "CONVERSATIONAL":
        plan = make_conversational_plan(user_message)
        steps.append("BUILDING CONTENT PLAN")

    elif message_type == "FACTUAL":
        retrieved = memory.retrieve(user_message)
        steps.append("SEARCHING MEMORY")
        retrieved = _augment_with_web_search(user_message, retrieved)
        # _augment_with_web_search already prints whether it searched — check the list
        steps.append("CHECKING SUFFICIENCY")
        if any(c.get("topic") == "web_search" for c in retrieved):
            steps.append("WEB SEARCH")
        plan = make_factual_plan(user_message, retrieved)
        steps.append("BUILDING CONTENT PLAN")

    else:  # HYBRID
        retrieved = memory.retrieve(user_message)
        steps.append("SEARCHING MEMORY")
        retrieved = _augment_with_web_search(user_message, retrieved)
        steps.append("CHECKING SUFFICIENCY")
        if any(c.get("topic") == "web_search" for c in retrieved):
            steps.append("WEB SEARCH")
        plan = make_hybrid_plan(user_message, retrieved)
        steps.append("BUILDING CONTENT PLAN")

    return {
        "content_plan": plan,
        "retrieved_chunks": retrieved,
        "message_type": message_type,
        "pipeline_steps": steps,
    }
```

### `src/agent.py` — pass steps through:

```python
def process_turn(memory, user_message, last_agent_response):
    if should_end(user_message, last_agent_response):
        signoff = add_signoff(last_agent_response)
        return {
            "response": signoff,
            "ended": True,
            "pipeline_steps": ["ENDING CONVERSATION"],
            "message_type": "END",
        }

    result = content_step(memory=memory, user_message=user_message)
    response = stylize(user_message=user_message, content_plan=result["content_plan"])

    steps = result["pipeline_steps"] + ["STYLING RESPONSE"]

    return {
        "response": response,
        "ended": False,
        "pipeline_steps": steps,
        "message_type": result["message_type"],
    }
```

### `src/api.py` — include in the HTTP response:

Update the `ChatResponse` Pydantic model:

```python
class ChatResponse(BaseModel):
    response: str
    ended: bool
    session_id: str
    pipeline_steps: list[str] = []
    message_type: str = ""
```

Update the `chat()` endpoint return:

```python
return ChatResponse(
    response=result["response"],
    ended=result["ended"],
    session_id=request.session_id,
    pipeline_steps=result.get("pipeline_steps", []),
    message_type=result.get("message_type", ""),
)
```

This is fully backward compatible — any client that ignores the new fields continues to work.

---

## 10. Welcome Screen

When the user first loads the page (no messages yet), the ChatArea should show a centered welcome state instead of an empty scroll area:

- Agent avatar icon (centered, 48x48)
- "WILLIAM RUTO" in Space Mono, uppercase, letter-spacing 0.1em, color `var(--color-text-secondary)`, font-size 13px
- "President of Kenya" in DM Sans, regular, color `var(--color-text-muted)`, font-size 14px
- 24px gap
- 3-4 clickable prompt chips arranged in a row (wrap on mobile):
  - "Tell me about your rise to presidency"
  - "What is your education background?"
  - "What are you working on currently?"
  - "What do you think about corruption?"
- Each chip: DM Sans 400, font-size 13px, color `var(--color-text-secondary)`, border 1px solid `var(--color-border)`, border-radius `var(--radius-pill)`, padding 8px 16px, cursor pointer
- On hover: background `var(--color-bg-subtle)`
- On click: populate the InputBar and auto-send

---

## 11. Conversation End State

When the backend returns `ended: true`:
- Show the agent's sign-off as the final message bubble
- Below it, show a centered "Conversation ended" label in Space Mono, uppercase, `var(--color-text-muted)`
- Disable the InputBar (gray out, show "Conversation ended" as placeholder)
- Show a "Start New Conversation" button (outlined, DM Sans 500, border-radius `var(--radius-pill)`) that calls `startNewChat()`

---

## 12. Error Handling

- **Network error / API timeout**: show a system message in the ChatArea: "Something went wrong. Please try again." with a "Retry" button that re-sends the last user message
- **404 session not found** (e.g., server restarted): automatically call `startNewChat()` and show a subtle toast: "Session expired — started a new conversation"
- **Empty response**: treat as an error, show "The agent didn't respond. Please try again."

---

## 13. Responsive Design

| Breakpoint | Layout |
|---|---|
| >= 1024px (desktop) | Two-column: ThinkingPipeline (380px) + ChatArea (fills remaining) side by side |
| 768px - 1023px (tablet) | Single-column: ThinkingPipeline collapses into a toggleable panel above ChatArea |
| < 768px (mobile) | Single-column: ThinkingPipeline becomes a compact inline element above the agent's response bubble (just the step labels as a single line with dots) |

Mobile-specific:
- Input bar: reduce padding, make send button 32px
- Message bubbles: max-width 90% instead of 70%
- Welcome prompt chips: stack vertically instead of horizontal row

---

## 14. Deployment

### Option A — Vercel (recommended, simplest)

1. Push the repo to GitHub (the `frontend/` directory is inside the existing repo)
2. Go to vercel.com → Import Project → select the repo
3. Set:
   - **Root directory**: `frontend`
   - **Framework preset**: Vite
   - **Build command**: `npm run build`
   - **Output directory**: `dist`
4. Add environment variable: `VITE_API_URL` = `https://your-render-app.onrender.com`
5. Deploy

---

## 15. Build Order (for the implementer)

Follow this exact sequence. Each step is independently testable before moving on.

1. **Scaffold** the Vite + React project inside `frontend/`
2. **Set up design tokens** (`tokens.css`, `index.css`, Google Fonts link)
3. **Build `Header.jsx`** — static, no logic
4. **Build `InputBar.jsx`** — text input + send button, wired to a local `onSend` callback
5. **Build `MessageBubble.jsx`** — both user and agent variants, with hardcoded test messages
6. **Build `ChatArea.jsx`** — scrollable list of `MessageBubble` components, auto-scroll, welcome screen
7. **Wire up `useChat.js`** — session management, API calls, message state (test with the real backend running locally via `uvicorn src.api:app --port 8000`)
8. **Build `ThinkingPipeline.jsx`** — the animated step list, driven by `useChat`'s `pipelineSteps`
9. **Apply backend modifications** from Section 9 (return pipeline metadata)
10. **Connect pipeline to real metadata** — use the `pipeline_steps` from the API response to validate/correct the animated pipeline
11. **Add conversation end state** (Section 11)
12. **Add error handling** (Section 12)
13. **Responsive breakpoints** (Section 13)
14. **Deploy** (Section 14)

---

## 16. CSS Animation Reference

### Pulse animation for active pipeline step:
```css
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(1.3); }
}

.stepActive .indicator {
  animation: pulse 1.5s ease-in-out infinite;
}
```

### Fade-in for new pipeline steps:
```css
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.step {
  animation: fadeInUp 0.3s ease-out forwards;
}
```

### Growing vertical line:
```css
.pipelineLine {
  width: 2px;
  background: var(--color-pipeline-line);
  transition: height 0.4s ease-out;
}
```

### Message bubble entrance:
```css
@keyframes slideIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

.bubble {
  animation: slideIn 0.25s ease-out;
}
```

---

*This plan is self-contained. Every design decision, file path, component spec, and API contract is included. Follow the build order in Section 15 sequentially.*
