# Bible-QA

## Abstract
<img width="919" alt="image" src="https://user-images.githubusercontent.com/73162197/185844960-12da6174-e834-4b45-8e1d-46c37c33b1c1.png">
- When user throws a question about bible info, fine-tuned Electra model tries to search for the best answer with using tf-idf similary search for short time search.

## TF-IDF similarity search
<img width="1060" alt="image" src="https://user-images.githubusercontent.com/73162197/185847533-524b258e-a511-45b5-b6e8-1de5950b89b9.png">

1. Vectorize chatpers and question with td-idf.
2. Get top 20 chapters that are similar to question using cosine-similarity.
3. Insert the whole 20 chapters as one long text and question to the language model.


## Long-Text QA

### Problem 1

<img width="718" alt="image" src="https://user-images.githubusercontent.com/73162197/186059638-4de847e4-db2a-48de-852a-405ef6914a68.png">

1. Original ElectraForQuestionAnswering Model only allows maximum length of 512 tokens as input.
2. If the length of text and question exceeds the maximum length, inference is impossible.

### Solution 1

<img width="705" alt="image" src="https://user-images.githubusercontent.com/73162197/186061645-b3f6fe13-930b-42f7-aec9-88445d334b08.png">
