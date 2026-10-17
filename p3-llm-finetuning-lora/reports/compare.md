# Base vs tuned (LoRA)

Test set n=50, greedy, max_new_tokens 200. ROUGE-L is stemmed F1 against the MedQuAD reference answer: it measures overlap with one reference text, not medical correctness.

| metric | base | tuned |
|---|---|---|
| ROUGE-L | 0.157 | 0.154 |
| avg words (ref 113) | 140 | 87 |

Paired per-question diff (tuned - base): mean -0.002, 95% bootstrap CI [-0.014, +0.011]; tuned wins 23, ties 0, losses 27.

Sanity: baseline.json ROUGE-L 0.157 (separate unbatched-adapter run) vs base here 0.157.

## Forgetting check (10 general prompts)

- ROUGE-L of tuned answers vs base answers: 0.398 (1.0 = unchanged)
- Keyword-correct on 4 checkable prompts (Paris, 102, Shakespeare, buenos): base 4/4, tuned 4/4
- avg words: base 78, tuned 28

| prompt | base | tuned |
|---|---|---|
| What is the capital of France? | The capital of France is Paris. | The capital of France is Paris. |
| Explain what a binary search does in two sentences. | A binary search algorithm searches for a specific value within an ordered list by repeatedly dividing the range of possible locations to find the target element... | A binary search is an algorithm that searches for a specific value within a sorted list of data. |
| Write a Python function that reverses a string. | Certainly! Below is a simple Python function that takes a string as input and returns the reversed version of that string: ```python def reverse_string(input_st... | def reverse_string(s): return s[::-1] |
| What is 17 multiplied by 6? | To calculate 17 multiplied by 6: \[ 17 \times 6 = 102 \] So, the result of 17 multiplied by 6 is 102. | To multiply 17 by 6, you can use the following steps: Step 1: Multiply 17 by 6. 17 x 6 = 102 Step 2: Add the result to 100. 102 + 100 = 202 Therefore, 17 multip... |
| Who wrote the play Romeo and Juliet? | Romeo and Juliet was written by William Shakespeare. He lived from 1564 to 1616 and is considered one of the greatest playwrights in the English language. The p... | The play was written by William Shakespeare. |
| Translate "good morning" into Spanish. | The translation of "good morning" into Spanish is "buenos días". | Good morning in Spanish is "buenos días". |
| Why is the sky blue? | The sky appears blue because of the scattering of sunlight by tiny particles in the Earth's atmosphere. Here’s how it works: 1. **Scattering**: When sunlight en... | The sky appears blue because of how light bounces off objects in the atmosphere. The Earth's surface reflects sunlight into space, but it also absorbs some of t... |
| Give me three tips for a job interview. | Certainly! Here are three effective tips to help you prepare for your job interview: 1. **Research the Company**: Before your interview day, research the compan... | 1. Prepare your resume and cover letter in advance. 2. Dress appropriately for the interview. 3. Be yourself. |
| Summarize the plot of Cinderella in two sentences. | Cinderella is a fairy tale about a young girl named Cinderella who lives with her cruel stepmother and older sister. She dreams of becoming a princess and falls... | Cinderella is a young girl who lives with her stepmother, older sister, and brother. She dreams of becoming a princess and wishes to marry the prince she loves.... |
| What is the difference between a list and a tuple in Python? | In Python, both lists and tuples are used to store collections of items, but they have some key differences: 1. **List**: - A list is an ordered collection of e... | In Python, a list is an ordered collection of items that can be accessed by index number. A tuple is an unordered collection of items that cannot be accessed by... |

## Qualitative (10 fixed medical prompts, side by side)

| prompt | base | tuned |
|---|---|---|
| What are the symptoms of type 2 diabetes? | Type 2 diabetes is characterized by high blood sugar levels in the blood. Here are some common symptoms that can indicate you might have Type 2 diabetes: 1. **Increased thirst and frequent urination**: This is often the first symptom many people notice when th... | Type 2 diabetes is characterized by high blood sugar levels that can cause damage to your body's organs and tissues. The most common symptoms include: - Feeling tired all the time - Having frequent urination - Having a dry mouth - Having a sore throat - Having... |
| How is high blood pressure diagnosed? | High blood pressure, also known as hypertension, is typically diagnosed through a combination of symptoms and possibly other tests. Here’s how it might be diagnosed: 1. **Symptoms**: The most common symptom of high blood pressure is headache, which can occur a... | High blood pressure (hypertension) is usually diagnosed by your doctor during a physical exam. Your doctor will ask about your symptoms and health history. He or she may also order tests to help diagnose the cause of your high blood pressure. These tests inclu... |
| What causes iron deficiency anemia? | Iron deficiency anemia is caused by a lack of iron in the body. This can happen for various reasons: 1. **Poor Iron Absorption**: The body may have trouble absorbing enough iron from food or supplements. 2. **Deficiency of Vitamin B12**: In some cases, vitamin... | Iron deficiency anemia is caused by a lack of iron in the body. Iron is needed to make hemoglobin, which carries oxygen throughout the body. Hemoglobin also helps carry oxygen from the lungs to all parts of the body. Without enough iron, your body can't make e... |
| What are the treatments for asthma? | Asthma is a chronic respiratory condition that can be effectively managed with various treatments. Here's an overview of some common approaches: 1. **Medications**: The most common medications used to treat asthma include: - Inhaled corticosteroids (ICS): Thes... | There is no cure for asthma. However, there are several medications that can help control symptoms and prevent attacks. These include bronchodilators (which relax the muscles in your airways), corticosteroids (which reduce inflammation), and long-acting beta a... |
| Who is at risk for osteoporosis? | Osteoporosis is a condition characterized by the weakening of bones due to a deficiency in bone density. It can affect anyone, but certain groups are more susceptible than others: 1. **Women**: Women are particularly vulnerable because their bodies lose calciu... | Osteoporosis occurs when bones become weak and fragile. It can happen to anyone, but it's more common in older women than men. Women who have had their ovaries removed before they're 40 years old are also at higher risk of developing osteoporosis. People with ... |
| What is glaucoma and how is it treated? | Glaucoma is a group of eye diseases that damage the optic nerve, which is responsible for transmitting visual signals from the retina to the brain. The most common type of glaucoma is primary open-angle glaucoma, which occurs in people without any other risk f... | Glaucoma is a group of eye diseases that damage the optic nerve. The optic nerve carries messages from the eyes to the brain. Damage to this nerve can cause vision loss. Glaucoma is usually diagnosed when someone has difficulty seeing things clearly. It's impo... |
| How can I prevent kidney stones? | Preventing kidney stones is important for maintaining kidney health and preventing complications such as urinary tract infections (UTIs) and chronic kidney disease. Here are some effective strategies to help reduce your risk: 1. **Dietary Changes**: - Increase... | The best way to prevent kidney stones is to eat a healthy diet rich in fruits, vegetables, whole grains, lean meats, and low-fat dairy products. The American College of Rheumatology recommends that people with kidney stones take calcium supplements daily. Thes... |
| What are the complications of untreated hypothyroidism? | Untreated hypothyroidism can lead to several serious complications if left unchecked: 1. **Cardiovascular Disease**: Hypothyroidism is associated with an increased risk of cardiovascular disease due to its effects on the heart muscle. 2. **Neurological Complic... | Untreated hypothyroidism can lead to serious health problems. These include: • Heart disease • Kidney failure • Muscle weakness • Weight gain • Enlarged thyroid gland • Thyroid cancer |
| How is celiac disease diagnosed? | Celiac disease is typically diagnosed through a combination of symptoms and certain tests: 1. **Symptoms**: The most common symptom is abdominal pain and diarrhea that can be severe enough to cause vomiting. Other symptoms include fatigue, weight loss, and ane... | Celiac disease is diagnosed by a combination of symptoms and tests. The most common symptom is abdominal pain and diarrhea that may be accompanied by vomiting and weight loss. Other symptoms include fatigue, bloating, gas, and constipation. A blood test called... |
| What is the outlook for people with Parkinson disease? | Parkinson's disease is a progressive neurological disorder that affects movement control in the body. The outlook for individuals with Parkinson's can vary widely depending on several factors: 1. **Age of onset**: People who develop Parkinson's later in life t... | The outlook for people with Parkinson's disease depends on how severe their condition is and what treatments they use. The outlook can vary greatly from person to person. Some people with Parkinson's disease have no symptoms at all. Others may experience mild ... |

## Interpretation (written after the run; numbers above are from compare.json)

- No measurable ROUGE-L gain: 0.157 base vs 0.154 tuned, paired diff -0.002 with a 95% CI spanning zero (23 wins / 27 losses on n=50). With 300 training examples and 37 steps this is expected; the result is neutral, not positive.
- What did change is style: tuned answers are shorter (87 vs 140 words; reference averages 113) and read like MedQuAD prose (plain paragraphs, "your doctor") instead of the base model's markdown lists. This is the usual first effect of small-data SFT.
- Qualitatively the tuned model sometimes drifts into generic or wrong detail (e.g. type 2 diabetes symptom list includes "sore throat"); ROUGE does not measure correctness and no clinician reviewed outputs.
- Forgetting: all 4 keyword-checkable general prompts still contain the right answer (4/4 both), but general answers shrank from 78 to 28 words on average and one tuned arithmetic answer showed bad reasoning (17 x 6 = 102, then "add 100 = 202"). Mild style drift plus some reasoning degradation; n=10, anecdotal.
- Verdict: LoRA adapted the tone/length to the domain without a quantitative quality gain at this scale. Next levers (not run): more data (standard profile), more epochs, lower lr, better eval (LLM judge or clinician review).
