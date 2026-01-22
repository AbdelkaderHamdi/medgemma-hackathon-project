# Clinical Documentation Assistant (MedGemma)

## Overview
This project is a hackathon demo that shows how open-weight medical AI models
(MedGemma) can be used to automatically generate clinical documentation.

The application helps doctors convert short clinical notes into structured
medical reports while running fully offline for privacy and reliability.

---

## Problem
Doctors spend a significant amount of time writing repetitive clinical notes.
This reduces time with patients and increases fatigue and errors.

Many clinical environments cannot rely on cloud-based AI due to:
- Privacy constraints
- Limited internet access
- Regulatory requirements

---

## Solution
An offline clinical documentation assistant that:
- Accepts short medical notes
- Generates structured medical reports
- Runs locally using MedGemma
- Keeps all data private

---

## Tech Stack
- **Frontend:** React (simple UI)
- **Backend:** Python + FastAPI
- **AI Model:** MedGemma (local inference)
- **Deployment:** Docker (CPU-only)

---

## Project Structure
```

frontend/   → User interface
backend/    → API and business logic
model/      → AI inference and prompts
docs/       → Product & AI design documentation
demo/       → Screenshots and demo video

```

---

## How It Works
1. Doctor enters clinical notes in the UI
2. Frontend sends notes to backend API
3. Backend calls MedGemma locally
4. AI generates a structured medical report
5. Report is returned and displayed

---

## Privacy & Ethics
- Runs fully offline
- No patient data is stored
- AI assists documentation only
- Final review is always done by a clinician

---

## Disclaimer
This tool does not provide medical diagnosis or treatment decisions.
It is intended only to assist with clinical documentation.
