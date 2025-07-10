Buffett Agent
=============

This tutorial demonstrates how to fine-tune a financial advisor model that mimics Warren Buffett using the FinLoRA framework [FinLoRA]_. The complete process costs less than $20 and showcases a practical application of fine-tuning LLMs for financial advisory tasks.

Motivation
----------

After spending weeks researching a company, many investors remain convinced it's a winner but still harbor that flicker of doubt... are they missing something only a seasoned expert would see? Hiring a world-class analyst like Warren Buffett to vet investment decisions would cost a fortune. However, it's possible to build a personalized financial advisor, a Buffett Agent, for a cost of $20 or less.

This tutorial is structured into three main parts:

1. **Building the Buffett Brain**: The process of creating the dataset and using this data to fine-tune the model
2. **Head-to-Head with Other Models**: Testing the model against other LLMs with practical questions and outlining ideas for future improvements  
3. **Putting Buffett Agent to the Test**: Analyzing the performance of the agent against base Llama 3.1 8B Instruct on a portion of the test set

Building the Buffett Brain
--------------------------

LLMs are only as good as their underlying data. To create an agent that mimics Warren Buffett in his authentic voice, the process involves searching online for content related to Warren Buffett and his company, Berkshire Hathaway, then curating a massive dataset of 25,000+ question-answer pairs from a wide range of sources.

Data Sources and Questions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The dataset is created from the data sources shown in the table below after converting them to text files and cleaning them of artifacts like page numbers.

.. list-table:: Buffett Agent dataset statistics.
   :header-rows: 1
   :widths: 15 25 10 50

   * - Datasets
     - #Train/#Val/#Test
     - Total
     - Sources
   * - **Primary Sources (Total: 25.0k)**
     - 
     - 
     - 
   * - Wikipedia
     - 1.8k/227/229
     - 2.28k
     - Wikipedia
   * - Books
     - 2.0k/250/250
     - 2.5k
     - Warren Buffett Books
   * - Letters
     - 2.0k/250/250
     - 2.5k
     - Berkshire Hathaway Shareholder Letters
   * - Transcripts
     - 2.2k/275/276
     - 2.75k
     - Berkshire Hathaway Shareholder Meeting Transcripts
   * - 10-K Filings
     - 4.0k/500/501
     - 5.0k
     - Berkshire Hathaway 10-K filings
   * - 10-Q Filings
     - 8.0k/996/998
     - 9.97k
     - Berkshire Hathaway 10-Q Filings
   * - **Total**
     - **20.0k/2.5k/2.5k**
     - **25.0k**
     - 

**Wikipedia**: The Wikipedia pages used are about Warren Buffett as well as information related to his investments and strategies. They provide high‑level facts on Warren Buffett, Berkshire Hathaway, and Warren Buffett's investment strategies to ground the Buffett Agent's basic knowledge. The pages are `Warren Buffett <https://en.wikipedia.org/w/index.php?title=Warren_Buffett&oldid=1293799158>`_, `Berkshire Hathaway <https://en.wikipedia.org/w/index.php?title=Berkshire_Hathaway&oldid=1295291763>`_, `Berkshire Indicator <https://en.wikipedia.org/w/index.php?title=Buffett_indicator&oldid=1266302658>`_, `List of Berkshire Hathaway Publications <https://en.wikipedia.org/w/index.php?title=List_of_Berkshire_Hathaway_publications&oldid=1287885798>`_, and `Wilshire 5000 <https://en.wikipedia.org/w/index.php?title=Wilshire_5000&oldid=1292526933>`_.

**Books**: The questions in this dataset are from the books `The Essays of Warren Buffett <https://www.amazon.com/Essays-Warren-Buffett-Lessons-Corporate/dp/1611637589>`_, `The Snowball <https://www.amazon.com/Snowball-Warren-Buffett-Business-Life/dp/0553384619>`_, and `The Warren Buffett Way <https://www.amazon.com/Warren-Buffett-Way-Robert-Hagstrom/dp/1118503252>`_. These questions are about Buffett's life story, investment decisions, and his investment philosophy.

**Letters**: The letters portion of the Buffett Agent data is based on Berkshire Hathaway's annual shareholder letters. These shareholder letters are written by Warren Buffett and detail his investing advice and philosophy, Berkshire Hathaway business decisions, and Berkshire Hathaway performance. They help capture Warren Buffett's authentic voice and thoughts. Berkshire Hathaway shareholder letters can be found on `Berkshire Hathaway's website <https://www.berkshirehathaway.com/letters/letters.html>`_.

**Transcripts**: The transcripts portion of the dataset comes from Berkshire Hathaway's annual meetings. These meetings contain Q&A sessions where Warren Buffett responds to investors' questions and concerns in a conversational tone. The dataset uses transcripts from the 1994–2008 and 2025 annual shareholder meetings. The transcripts for 1994–2018 can be found `here <https://sorfis.com/wp-content/uploads/2021/09/1-Berkshire-Transcripts-1994-2018.pdf>`_, and the 2025 transcript can be found `here <https://steadycompounding.com/transcript/brk-2025/>`_.

**10-K, 10-Q Filings**: The Buffett Agent 10-K and 10-Q data is based on Berkshire Hathaway's SEC filings. They outline Berkshire Hathaway's revenue and performance. This dataset helps the Buffett Agent deal with more technical questions related to Berkshire Hathaway. The dataset uses 10-Q filings from 1996–2000 and 2020–2024. The 10-K filings are from 1996–2024. The filings can be found on `Berkshire Hathaway's website <https://www.berkshirehathaway.com/reports.html>`_.

Data Processing Steps
^^^^^^^^^^^^^^^^^^^^^

**Step 1: Chunking**

After converting the data to text files, create chunks of words to pass as context to GPT-4o mini for creating Q&A pairs. Since these chunks are passed as part of prompts to GPT-4o mini, tokenization is not required. According to `OpenAI <https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them>`_, a paragraph is around 75 words. To capture enough relevant information while generating the target amount of meaningful questions, use chunks of 300 words (the last chunk could contain less), which is around 4 paragraphs of content. Also use a small overlap window (20%) to ensure context isn't cut off inappropriately (e.g., a question asked to Warren Buffett in a shareholder meeting isn't in a separate chunk from his answer). Padding is not used because the chunk is passed directly to GPT-4o mini as part of a prompt.

.. code-block:: python

   def chunk_passage(text, word_count=300):
       words = text.split()
       overlap = int(word_count * 0.20)   # 20% overlap
       step = word_count - overlap        # move forward 80%
       chunks = []

       for i in range(0, len(words), step):
           segment = words[i : i + word_count]
           chunks.append(" ".join(segment))

       return chunks

**Step 2: Generating Q&A Pairs**

After obtaining the data and chunking it, convert it into realistic conversational Q&A pairs by prompting GPT-4o mini to use the context to generate authentic Q&A pairs based on those chunks. 

Use the following prompt with GPT-4o mini to create these pairs:

.. code-block:: text

   Generate Warren‑Buffett–style Q&A pairs for fine-tuning a Buffett Agent.

   Context:
   <INSERT_CONTEXT_HERE>

   Task:
   1. Read the passage.
   2. Write exactly three question‑answer pairs.

   Guidelines:
   • Each question speaks directly to Buffett and references a concrete fact, figure, or theme from the passage.  
   • Each answer is Buffett replying in first person with his plain, folksy, and practical tone.

   Output format (one line per pair, no extra blank lines):
   PAIR1: <Question>|<Answer>
   PAIR2: <Question>|<Answer>
   PAIR3: <Question>|<Answer>

   If the passage contains no meaningful business or investing content, output nothing.

   Example:
   PAIR_EXAMPLE: In the 2025 Q1 report you held about $300 billion—roughly 27% of assets—in cash. Why keep so much dry powder when markets look pricey?|Well, I always say cash is like oxygen—when you need it and don't have it, nothing else matters. I'll gladly spend $100 billion tomorrow if the right opportunity appears, but fat pitches don't arrive on a timetable, so patience keeps us ready.

Re-generate questions to replace duplicate questions whenever GPT-4o mini creates the same Q&A pair. Normalize the Q&A pairs and store them in .jsonl file with "context" and "target" fields. The "context" field poses a question asked to Warren Buffett, and the "target" field contains his answer.

**Step 3: Train-Validation-Test Split**

Split the questions in the .jsonl files into 80% train, 10% validation, and 10% test:

.. code-block:: python

   import random

   with open("buffett_letters_qa.jsonl", "r", encoding="utf-8") as f:
       lines = f.read().splitlines()

   # Use a fixed seed when shuffling for reproducibility
   random.seed(42)
   random.shuffle(lines)

   n = len(lines)
   n_train = int(0.8 * n)
   n_val = int(0.1 * n)

   train_lines = lines[:n_train]
   val_lines = lines[n_train:n_train + n_val]
   test_lines = lines[n_train + n_val:]

   # Create the split files
   with open("buffett_letters_qa_train.jsonl", "w", encoding="utf-8") as f:
       f.write("\n".join(train_lines) + "\n")
   with open("buffett_letters_qa_validation.jsonl", "w", encoding="utf-8") as f:
       f.write("\n".join(val_lines) + "\n")
   with open("buffett_letters_qa_test.jsonl", "w", encoding="utf-8") as f:
       f.write("\n".join(test_lines) + "\n")

For the Buffett Agent, combine all the train files into one .jsonl file called ``buffett_agent_train.jsonl``.

Fine-tuning with FinLoRA
------------------------

With the dataset prepared, the final step is to bring the Buffett Agent to life using FinLoRA, a powerful framework designed for fine-tuning LLMs on financial datasets. It uses the efficient Low-Rank Adaptation (LoRA) [LoRA]_ technique to fine-tune LLMs at a much lower cost of compute and fine-tune time. For fine-tuning Buffett Agent, use Llama 3.1 8B Instruct as the base model.

Environment Setup
^^^^^^^^^^^^^^^^^

**1. GPU Requirements**

- CUDA: version ≥ 11.8
- VRAM (for Llama‑3.1‑8B Instruct):
  - ≥ 24 GB for 8‑bit quantization
  - ≥ 16 GB for 4‑bit quantization

**2. RunPod Cloud GPUs (Optional)**

If you lack a suitable local GPU, you can rent A5000s on RunPod:

a. In "Billing," add $10 of credits.
b. Create a Network Volume (under "Storage") with the following properties:
   - Size: 50 GB (~$3.50/mo)
   - Datacenter: choose one with A5000 availability.
c. Deploy Pod:
   - Template: "Runpod PyTorch 2.8.0"
   - GPU Count: 4 (or 1 for inference-only)
   - Pricing: on‑demand (~$0.26/hr per A5000)

**3. Package Installation**

You can either use the provided setup.sh or a conda environment.

**a. Using setup.sh**

The following installs packages using a bash script:

.. code-block:: bash

   git clone https://github.com/Open-Finance-Lab/FinLoRA.git
   cd FinLoRA
   # Make and run the installer
   chmod +x setup.sh
   ./setup.sh

**b. Using a Conda Environment**

To setup a Conda environment, execute the following commands:

.. code-block:: bash

   git clone https://github.com/Open-Finance-Lab/FinLoRA.git
   cd FinLoRA
   conda env create -f environment.yml
   conda activate finenv

**4. Login to Hugging Face**

Because Llama models are gated, you must authenticate before downloading or pushing:

.. code-block:: bash

   huggingface-cli login
   # Enter your Hugging Face token when prompted

Fine-Tuning Pipeline
^^^^^^^^^^^^^^^^^^^^

**1. Fetch Deepspeed Configs**

.. code-block:: bash

   cd lora
   axolotl fetch deepspeed_configs

The DeepSpeed configs will later be used to parallelize fine-tuning across GPUs.

**2. Add the Fine-Tuning Dataset**

Place the fine-tuning train file (e.g., ``warren_buffett_train.jsonl``) in ``FinLoRA/data/train/``.

**3. Configure your LoRA hyperparameters**

Open ``finetune_configs.json`` in the lora folder and add a block for your adapter. The one used follows:

.. code-block:: json

   "buffett_agent_llama_3_1_8b_8bits_r8_lora": {
     "base_model":  "meta-llama/Llama-3.1-8B-Instruct",
     "dataset_path":"../data/train/warren_buffett_train.jsonl",
     "lora_r":      8,
     "quant_bits":  8,
     "learning_rate": 1e-4,
     "num_epochs":  4,
     "batch_size":  2,
     "gradient_accumulation_steps": 2
   }

**4. Run Fine-Tuning**

.. code-block:: bash

   python finetune.py buffett_agent_llama_3_1_8b_8bits_r8_lora

This script parallelizes fine-tuning across GPUs via DeepSpeed. Checkpoints and the final adapter will be saved to ``lora/axolotl-output/<config-name>/``.

**5. Locate the Adapter** 

Locate the Adapter under ``FinLoRA/lora/axolotl-output/<adapter-name>``. The adapter path will be ``FinLoRA/lora/axolotl-output/buffett_agent_llama_3_1_8b_8bits_r8_lora/``.

**Training Environment and Cost**

The experiments can be run with the following environment:

- GPUs: 4 × RTX A5000 (96 GB total VRAM)
- RAM: 200 GB  
- vCPUs: 36
- Total Disk: 110 GB
- Network Volume: 70 GB
- Estimated Cost: $2.10

Using the LoRA Adapter for Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import PeftModel
   import torch

   # Load base model and tokenizer
   base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
   tokenizer = AutoTokenizer.from_pretrained(base_model_name)
   base_model = AutoModelForCausalLM.from_pretrained(
       base_model_name,
       torch_dtype=torch.float16,
       device_map="auto",
       trust_remote_code=True
   )

   # Load and apply the LoRA adapter
   adapter_path = "./lora/axolotl-output/buffett_agent_llama_3_1_8b_8bits_r8_lora"  # Path to adapter
   model = PeftModel.from_pretrained(base_model, adapter_path)

   # Generate text
   prompt = "What is the formula for the Black-Scholes model?"
   inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

   with torch.no_grad():
       outputs = model.generate(
           **inputs,
           max_new_tokens=1024,
           # This ensures reproducible responses
           temperature=0,
           pad_token_id=tokenizer.eos_token_id
       )

   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print(response)

Head‑to‑Head with Other Models
------------------------------

To evaluate the Buffett Agent, test it against other LLMs with a practical question. The same question can be asked to the Buffett Agent, the "Invest Like Warren Buffett" AI custom GPT, and GPT-4o. Here's an example question and the model responses:

**Question:**
"Please give a concise response like Warren Buffett: Given today's high‑cost capital environment, rapid technological shifts, and geopolitical headwinds, what disciplined framework should executives follow when deciding to reinvest in core operations, pursue bolt‑on acquisitions, execute share repurchases, or pay dividends — so as to protect the company's margin of safety and drive lasting per‑share value?"

**Buffett Agent Response:**
"You see, my friend, it all comes down to one simple principle: prioritize cash flow over growth. I always say that if you can't generate enough cash to cover your costs and still have some left over for reinvestment, then you're playing with fire. So, I'd advise focusing on maintaining a strong balance sheet, ensuring we've got a solid margin of safety, and only making investments that truly enhance our long-term value."

**ChatGPT (Invest Like Warren BuffettAI GPT with GPT-4o Base Model):**
"Warren Buffett-style concise response: In a world of high capital costs, rapid change, and geopolitical risk, stick to the basics: Reinvest only where returns clearly exceed the cost of capital; buy bolt-on acquisitions when you gain durable advantages at a fair price; repurchase shares only when they're well below intrinsic value; and pay dividends if no better use exists. Above all, protect your margin of safety and focus on long-term per-share value growth — not headlines."

**ChatGPT (GPT-4o):**
"Focus on what you know and control. Reinvest in core operations only where returns exceed the cost of capital by a wide margin. Pursue bolt-on acquisitions that strengthen your moat and are easy to integrate. Repurchase shares only when they're clearly undervalued. Pay dividends when you can't do better for shareholders with the money. Always protect cash, stay rational, and ignore the noise."

**Analysis**

The Buffett Agent already nails the "short, folksy, conversational" tone, but, unlike the Invest Like Warren BuffettAI GPT, it fails to mention some of Buffett's core investing principles — such as bolt-on acquisitions and dividends. GPT‑4o, meanwhile, offers solid financial advice but sounds more like it summarized a textbook. The gap between Buffett Agent and the custom GPT exists because the fine-tuning data focused on Warren Buffett's tone in responding rather than his terminology and investing principles. To close it, future improvements could enrich the fine-tuning data with passages in which Buffett explicitly uses his signature terms and his core investing principles. This could be done by revising the prompt used with GPT-4o mini, so that each generated Q&A pair includes one signature Buffett investing principle per answer. Invest Like Warren BuffettAI GPT's advantage in Buffett's terminology and principles came from it drawing on its knowledge and the documents that were used to create it. This is possible because of custom GPTs' abilities to use retrieval augmented generation (RAG) and tool-calling. A similar approach could be taken by connecting Buffett Agent to MCP servers — potentially in a deep research framework — so it can pull recent Berkshire letters, headlines, and filings on demand. With those upgrades, Buffett Agent will keep its authentic Buffett tone while adding the depth and vocabulary that make readers think that they're talking to the Oracle of Omaha himself.

Putting Buffett Agent to the Test
---------------------------------
The following evaluation tested Llama 3.1 8B Instruct (the base model) and the fine-tuned Buffett Agent on 20% of the test set (501 questions), which were randomly selected with seed 42.

The responses were scored with BERTScore [BERTScore]_. BERTScore is an automatic evaluation metric for text generation. It represents reference and candidate sentences as embeddings. BERTScore measures the similarity between the embeddings for reference and candidate sentences using cosine similarity. Each token in the reference sentence is matched with the most similar token in the candidate sentence to compute recall. The opposite occurs to compute precision. These are used to calculate an F1 score.

The average BERTScore F1 for the base model and Buffett Agent is shown below:

.. image:: /_static/images/Buffett_Agent_testing.png
   :alt: BERTScore F1 comparison showing Buffett Agent improvement
   :align: center

The fine-tuned Buffett Agent shows a clear improvement of around 8.8% in BERTScore F1, highlighting the efficacy of LoRA as an effective yet affordable fine-tuning technique. The responses given more closely align with the verbiage of the ground truth answers.

Buffett Agent Links
-------------------

- `Buffett Agent Medium Blog <https://medium.com/finlora/build-my-buffett-agent-with-finlora-d27ff692b217>`_
- `Buffett Agent Data <https://huggingface.co/datasets/ghostof0days/Buffett_Agent_Data/settings>`_
- `Buffett Agent Model <https://huggingface.co/ghostof0days/buffett_agent_llama_3_1_8b_8bits_r8_lora>`_



Related Documentation
---------------------

For more information on the FinLoRA framework and related topics:

- :doc:`setup` - Environment setup and installation
- :doc:`finetune` - General fine-tuning procedures  
- :doc:`eval` - Evaluation methods and metrics
- :doc:`data_processing` - Dataset processing techniques
- :doc:`../lora_methods/vanilla_lora` - LoRA methodology details