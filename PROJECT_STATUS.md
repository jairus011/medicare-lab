# Project Status

## Purpose
Educational symptom-to-condition ML prototype built from the supplied healthcare project materials.

## What works
- audited dataset and source materials
- leakage-aware grouped evaluation
- reproducible training pipeline
- saved evaluated model artifacts
- Streamlit application
- automated tests
- browser screenshots
- verification scripts
- Render deployment

## Deployment
Live on Render:
https://medicare-lab.onrender.com

## API
No external API is integrated.

The Streamlit app loads the local evaluated artifact and source-grounded reference files. It does not call an external medical API or use an LLM to generate medical recommendations.

## Important model limitation
The selected model correctly predicts only 2 of 38 held-out records (5.26%). This is intentionally disclosed. The project demonstrates responsible evaluation and software engineering, not clinically useful diagnosis.

## Portfolio role
Strong software-engineering and responsible-ML evidence, but it should sit below the stronger banking/fraud projects because the model itself is not clinically useful.

## Next improvements
- keep the current limitation disclosure
- do not tune against the existing holdout
- only extend the modelling after obtaining substantially better governed data
- preserve deployment and test reproducibility
