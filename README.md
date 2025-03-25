# Within-Label Variation in Natural Language Inference: A Linguistic Taxonomy for Explanations and Its Impact on Model Interpretation of Label Decisions

This is a repository for my master thesis **Within-Label Variation in Natural Language Inference: A Linguistic Taxonomy for Explanations and Its Impact on Model Interpretation of Label Decisions**.

### Thesis Goals and Contributions
This thesis aims to investigate the linguistic characteristics of explanations in Natural Language Inference (NLI) and their impact on model interpretation of label decisions. Specifically, the objectives are:
- Develop a linguistic taxonomy for NLI explanations
- Improve NLI annotation and disagreement resolution (propose an annotation guideline to help annotators incorporate linguistic reasoning into their decision-making process)
- Analyze the impact of linguistic explanations on model interpretability
- Enhance LLM-generated explanations for NLI (diverse, linguistically grounded explanations that align with human reasoning)

### Annotation Process
The annotation interface is designed using Streamlit(https://github.com/streamlit/streamlit)

To run the interactive web app, please follow the Quickstart instruction provided on Streamlit Guide(https://github.com/streamlit/streamlit).

Then, open a terminal and run:
```bash
$ streamlit run explation_annotation.py
```

### Datasets and Preprocessing for Annotation

```bash
$ python reformat_esnli.py --input /your/esnli.csv --output /your/output.csv
```

```bash
$ python reformat_varierr.py --input /your/varierr.json --output /your/output.csv
```