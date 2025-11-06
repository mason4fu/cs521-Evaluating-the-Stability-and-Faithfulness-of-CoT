# CoT Stability (Minimal Setup)

This is a minimal setup for CoT stability experiment

### 🔑 Setup API Keys

Create a `.env` file in the project root with your own API key(s):
For Google Gemini
```
USE_GEMINI=true
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.5-flash
SAFETY_DEFAULT=BLOCK_MEDIUM_AND_ABOVE
```

To run the experiment,
```
sh run.sh
```
### Prof’s advice:
 - Apply for A100 GPU resources, deploy the LLM with more test cases.
 - Experiment with adversarial attacks to uncover interesting findings and evaluate robustness.
