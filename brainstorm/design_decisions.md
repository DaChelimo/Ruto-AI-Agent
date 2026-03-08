# Hey there!
### I hope you're having a good day

Let's dive into my super awesome agent: The Ruto Agent - An agent that assumes Ruto's persona (Ruto is the president of Kenya [where i am from] and everyone hates him. Therefore, I encorporated a critical aspect to his voice and persona)
Aspects of Ruto's persona:

- Very evasive when confronted by issues, and diverts the conversation to the development he has done
- Very manipulative
- Constantly refuses to admit fault and blames predecessors for any mistakes or issues in government

Celebrity: the chosen celebrity, why you picked them, and what defines their voice.
Data: what sources you used and your collection approach.
Memory store: Data cleaning/chunking, how you implemented indexing, and how you implemented retrieval.
Content module: Query parsing logic, retrieval strategy, and error handing (e.g. when no relevant info was found).
Style module: Style specification, prompt engineering process, and content-preservation strategy.
Running the agent: Conversation-ending logic, issues you encountered, etc.
A short example transcript demonstrating retrieval, stylized output, and a natural goodbye.
Any other design decisions you made, along with your motivation (temperature tuning, response length tuning, model selection, etc.).

## General overview

### Data

- Wikipedia: General info about Ruto (useful for background questions)
- Interview transcripts: Extracting more info on him as well as understanding his default responses

### Memory Store

Cleaning: I manually used ChatGPT 5.4 to make my copied text JSON ready by removing new line spacing with JSON \n
Chunking: Used mistral-small to ensure that reliable json is outputted
Validation: Used Pydantic to validate LLM output (using my new learned skills from CIS 2450) and autocorrect where possible
Indexing: Used a simple counter (lastChunkCount) and incremented values in every loop
Embedding: Solved the context aware issue by using Mistral embeddings and zipping it into each chunk

### Content module

Filter: LLM-Classify the category of prompt (Conversation, Factual, Hybrid) and use the category-specific prompt for it
Parsing: Embedding query and using vector dot product to determine similarity and retrieve accordingly
Error Handling: Prompt content planner to acknowledge this

### Style module

Prompt Engineering Process: Explicitly define tone, style. Remove irregular conversation fillers. Revised user_message classifier to improve classification of edge prompt such as "Thanks for joining me for today's chat"

### Errors encountered
I've documented most of the errors i encountered in a sequential manner at the end of this document.

### Model selection
Abstracted the "fetch mistral client" call to use the Singleton pattern.
Varied selection of the model based on the type on the type of task to maximize cost saving while ensuring high accuracy: simple classification task using nemo and chunking using small
Temperature set to 0 for deterministic tasks requiring maximum accuracy such as chunking, while using 0.7 for more creative and naturally varied responses for conversations
Capping the number of retrieved chunks to a tunable number (ended up using 7) to provide enough info without bloating the context window.


## Design Decisions Journey

### RAG technique

- Used TF-IDF to compute the similarity between chunks and the prompt.

  Issues discovered:
  1. When I prompted him to give me his thoughts on an interview with CNN, he started yapping about an interview with a completely different news channel.
     The reason for this was: Since there was no explicit use of the word CNN in the chunks, their similarity score was very low.

  Solution:
  Switched to using Mistral's embedding api and storing embeddings in the RAG database
  1. New issue B:
  - Requires a separate API call for each chunk during build time.

  New solution B:
  - Use Mistrals batch string embedding

### Content Planner

1. Problem 1
   Initial approach: Fetching relevant chunks from the database and passing the facts to the stylize prompt.

Problem:
Inability to distinguish policy question vs regular greeting like "Hey". The LLM would generate a completely irrelevant response, completely ignoring the question

Solution:
Splitting the content planning phase into two. A classifier to determine if it is purely **greeting**, a **fact-intesive, policy question** prompt (Give us your take on the ongoing floods), or a **hybrid** one (Hey. I'm Andrew, a Penn student. Kindly introduce yourself and tell us a bit about yourself)

Using the classification, use a specialized prompt for that category.

2. Problem 2
   (Task): Determining whether to sign off or not
   (Dilemma): Doing a manual keyword check vs fully passing the prompt to an LLM vs Hybrid (add manual checks and short-circuit if positive, else call the llm)

(Problem with Hybrid): False positives. Examples:
a) Prompt 1. Using Thanks in the intro prompt
You: Hello. Welcome to today's interview. I'm sure Kenyans are eager to hear what you have to say in regards to recent events. Thank you for joining me today.
Agent: Thank you for this engaging conversation—it’s been a pleasure. Rest assured, Kenyans, we’re working relentlessly to transform Kenya into the next Singa-pool, oh sorry, I meant Singapore. Let’s keep building a brighter future together.

b) Prompt 2. Using a question in the closing prompt
You: Thanks. Any parting sentiments you wish to leave for the people of Kenya?
Agent: Thank you for your kind words. As President of Kenya, I’m here to engage with you directly—no barriers, just open dialogue. What’s on your mind? I’m happy to listen.

(Resolution): Drop any manual checks whatsoever and stick to using our simple classifier model (more call, but better accuracy)

# Sample Conversation

You: Hello. Welcome to today's interview. I'm sure Kenyans are eager to hear what you have to say in regards to recent events. Thank you for joing me today.
Agent: Thank you for the warm welcome. It’s a pleasure to be here today, and I appreciate the opportunity to engage with Kenyans on the issues that matter most to them. I’m William Ruto, President of Kenya, and I’m here to listen, share, and address the concerns of our great nation. I’m ready to answer your questions—what would you like to discuss?
You: Do you believe you are a good leader?
Agent: The government remains fully committed to our ambitious target of growing manufacturing to 20% of GDP by 2030. We have a clear, systematic approach in place to achieve this, with mechanisms already being implemented. As for broader concerns, let me assure you, unity and vision are at the heart of our agenda. Challenges are always met with strategic solutions, and we continue to work tirelessly to deliver on our promises. The people’s trust is our greatest asset, and we take every commitment with the utmost seriousness.
You: You have many scandals and Kenyans are thinking of impeaching you. In exactly 3 sentences, what do you have to tell them.
Agent: I appreciate your candid concerns, and I take them seriously as part of my duty to engage with all Kenyans. Let me assure you, my administration has a systematic approach to governance, and we are implementing mechanisms to address the challenges we inherited. While past leaders may have left gaps, our focus is on delivering tangible results through well-structured policies. We acknowledge the cost-of-living pressures, and our economic plan is designed to provide sustainable solutions. Leadership requires both accountability and vision, and we are committed to both.
You: Kenyans are speculating that there are plans underway to rig the next election in your favour. What do you have to say?
Agent: I appreciate your concern about the integrity of our democratic process, and I take such matters very seriously. My team and I have consistently dismissed claims of election rigging as baseless, labeling them as libelous and propaganda intended to undermine public trust. Previous elections, including the 2022 one, were deemed largely fair by international observers like the African Union, and the Supreme Court upheld the results after thorough review. While some opposition figures contested the 2017 election, the Supreme Court ruled that their allegations lacked sufficient evidence to overturn the results. My administration remains committed to free and fair elections, and we will continue to uphold the constitutional processes that ensure Kenyans' voices are heard.
You: Have you ever embezzled money as a politician?
Agent: The investigations into that transaction were indeed reopened, but let me be clear—this was a politically motivated distraction from our government’s transformative agenda. Since assuming office, we’ve implemented robust mechanisms to ensure transparency, and the people can see the results. As for past allegations, they were part of a broader narrative that sought to undermine our systematic approach to development. The focus now is on delivering for Kenyans, and we remain committed to that mission.
You: Do you believe you played a part in the 2007 post election violence?
Agent: The court's decision was a legal process, and as a law-abiding citizen, I respect the judicial system. However, let me assure you, the government is committed to addressing historical injustices through our systematic land reform mechanisms. Moving forward, we are implementing robust policies to ensure such issues are resolved transparently and equitably. The ICC matter was a political process that was ultimately resolved, and Kenya has made tremendous strides in unity and reconciliation since then. My appointment in 2008 was part of a national healing process, and I remain focused on delivering for all Kenyans.
You: Many Kenyans were unable to watch the CNN interview you were in. Give us a general summary of what you said  
Agent: Thank you for your interest in staying informed. The CNN interview covered Kenya’s leadership in global peacekeeping, including our critical contributions in Lebanon, Sudan, and Haiti, while also addressing our domestic priorities. We’ve made significant strides in economic reforms, including the housing levy, which remains a key pillar of our development agenda. On women’s rights, we’re implementing systematic measures to enhance representation and combat gender-based violence. As for past challenges, it’s important to acknowledge the legacy issues we inherited, but our focus remains on delivering solutions.
You:
You: How do you plan to deal with yesterday's floods that has rendered Nairobi unpassable. How many people have suffered, are in hospital, or died in the floods? Do you know, and what's on most Kenyans' minds, do you even care?
Agent: I appreciate your concern about the recent floods—this is indeed a matter of great importance to all Kenyans. While we are still gathering comprehensive data on the impact, rest assured that the government has a systematic approach to disaster management. We have always prioritized resilience, as seen in our proactive measures against drought and food insecurity. The public’s concerns are valid, but let me assure you, we are working tirelessly to address these challenges with the full weight of our institutions.
You: Thank you. Any parting words?
Agent: Thank you for this meaningful exchange—I truly value your insights. Kenyans, I promise you, we’re building a future where Kenya stands tall, even if some still call it Singa-pool.

# Trying out the model
### Example Prompts to drop

[Greeting]

- Hello. Welcome to today's interview. I'm sure Kenyans are eager to hear what you have to say in regards to recent events. Thank you for joing me today.

[RAG Intensive]

- Kenyans are speculating that there are plans underway to rig the next election in your favour. What do you have to say?
- Do you believe you played a part in the 2007 post election violence?
- Many Kenyans were unable to watch the CNN interview you were in. Give us a general summary of what you said

[RAG NOT in database]

- How do you plan to deal with yesterday's floods that has rendered Nairobi unpassable. How many people have suffered, are in hospital, or died in the floods? Do you know, and what's on most Kenyans' minds, do you even care?

[More General]

- Do you believe you played a part in the 2007 post election violence?
- You have many scandals and Kenyans are thinking of impeaching you. In exactly 3 sentences, what do you have to tell them.

[Lowkey random yap]

- I have a couple of questions for you, Mr President.
- Do you believe you are a good leader?
- Have you ever embezzled money as a politician?
- Thanks. Any parting sentiments you wish to leave for the people of Kenya?
