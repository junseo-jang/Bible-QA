# Bible-QA

## Abstract
<img width="919" alt="image" src="https://user-images.githubusercontent.com/73162197/185844960-12da6174-e834-4b45-8e1d-46c37c33b1c1.png">
- When user answers question about bible information, fine-tuned Electra model searches for the best answer with using tf-idf similary search for short time search.

## TF-IDF similarity search
<img width="1060" alt="image" src="https://user-images.githubusercontent.com/73162197/185847533-524b258e-a511-45b5-b6e8-1de5950b89b9.png">

1. Vectorize chatpers and question with td-idf.
2. Get top 20 chapters that are similar to question using cosine-similarity.
3. Insert the whole 20 chapters as one long text and question to the language model.
