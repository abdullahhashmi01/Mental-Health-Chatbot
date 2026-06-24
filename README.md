# Mental Health Support Chatbot (Fine-Tuned DistilGPT2) — Industry-Style Structure

A modular, pipeline-based NLP project: fine-tunes DistilGPT2 on the
**EmpatheticDialogues** dataset to give gentle, supportive responses.
Structured like a real ML project — components, pipelines, config,
logging, and custom exceptions — instead of one big script.


---

## Project Structure

```
mental_health_chatbot_pro/
├── artifacts/                          # generated: raw data, transformed data, model
│   ├── data_ingestion/
│   ├── data_transformation/
│   └── model_trainer/
└── logs/                                 # generated: timestamped run logs
├── src/
│   ├── constants.py                    # single source of truth for config values
│   ├── logger.py                         # centralized logging -> logs/*.log
│   ├── exception.py                       # custom exception w/ file+line info
│   ├── entity/
│   │   └── config_entity.py                 # typed config dataclasses per stage
│   ├── components/
│   │   ├── data_ingestion.py                  # Stage 1: download raw dataset
│   │   ├── data_transformation.py              # Stage 2: clean + build pairs
│   │   ├── model_trainer.py                      # Stage 3: fine-tune DistilGPT2
│   │   └── safety_filter.py                       # crisis-keyword safety layer
│   └── pipeline/
│       ├── training_pipeline.py                     # chains stages 1->2->3
│       └── prediction_pipeline.py  
├── main.py                          # entry point -> runs training pipeline
├── chat_cli.py                       # CLI app (uses prediction pipeline)
├── app_streamlit.py                   # Streamlit web app (uses prediction pipeline)
├── setup.py
├── requirements.txt       
├──params.yaml
              
```

## Why this structure?

This mirrors common production ML repo layouts:

- **`components/`** — each pipeline stage is an independent, testable class
  with a single responsibility (ingestion ≠ transformation ≠ training).
- **`pipeline/`** — orchestrators that chain components together. `training_pipeline.py`
  is the only thing `main.py` calls; it doesn't know *how* each stage works internally.
- **`entity/config_entity.py`** — typed config objects instead of passing raw
  strings/dicts between functions. Easy to override for experiments.
- **`constants.py`** — all tunable values (paths, hyperparameters, prompts,
  safety keywords) live in one file, not scattered across scripts.
- **`logger.py` / `exception.py`** — every component logs progress to a
  timestamped file in `logs/`, and every error is wrapped with the exact
  file + line number where it happened (`ChatbotException`).
- **`safety_filter.py`** — isolated on purpose. For a mental-health-adjacent
  chatbot, the crisis-detection logic should be easy to find, test, and
  upgrade independently of the generation model.

## Setup

```bash
cd Mental-Health-Chatbot
python -m venv venv
source conda activate envir/      
pip install -r requirements.txt
```

> If `torch` install is slow: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Run the training pipeline (Ingestion -> Transformation -> Training)

```bash
python main.py
```

This will:
1. Download EmpatheticDialogues -> `artifacts/data_ingestion/`
2. Build clean (context, response) pairs -> `artifacts/data_transformation/`
3. Fine-tune DistilGPT2 -> `artifacts/model_trainer/distilgpt2-empathetic/`

All progress is logged to console and to a new file in `logs/`.

You can also run any single stage independently, e.g.:
```bash
python -m src.components.data_ingestion
python -m src.components.data_transformation
python -m src.components.model_trainer
```

**Tip for a quick sanity-check run first:** lower `MAX_TRAIN_EXAMPLES` and
`NUM_TRAIN_EPOCHS` in `src/constants.py` (e.g. to `1000` and `1`) before
doing a full run, so you can confirm everything works end-to-end on CPU
before committing to a longer training time.

## Test the chatbot

CLI:
```bash
python chat_cli.py
```

Streamlit web app:
```bash
streamlit run app_streamlit.py
```

## Extending this project

- Replace `SafetyFilter`'s keyword list with a trained classifier.
- Add a `model_evaluation.py` component (e.g. perplexity on test set) between
  training and deployment.
- Add a `config/config.yaml` + a `ConfigurationManager` class if you want
  config driven by YAML instead of `constants.py` (very common in real repos).
- Containerize with a `Dockerfile` and deploy the Streamlit app to Hugging
  Face Spaces or Streamlit Community Cloud.
