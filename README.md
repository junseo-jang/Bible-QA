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
1. Divide the input text so that each partition's length does not exceed the maximum input lenth of Electra.
2. Run the model for each partition and get the start, end position possibility value.
3. Pick the answer that has the highest (start_pos + end_pos)/2 value.

### Problem 2

<img width="781" alt="image" src="https://user-images.githubusercontent.com/73162197/186062468-000a9f43-6a41-4872-a435-be1759d1292e.png">

1. In the process of dividing text, a text can be divided in the middle of sentence.
2. This might lead to mischoosing the correct answer.

### Solution 2

<img width="712" alt="image" src="https://user-images.githubusercontent.com/73162197/186070347-64245938-8fda-41e9-af66-a69066dc8025.png">

1. Move the division boarder to the closest . or ? or ! behind.

### Improvement

<img width="629" alt="image" src="https://user-images.githubusercontent.com/73162197/186071687-b9f8fd7a-c440-4b94-b2d1-b11fe3d36d00.png">

1. Changing the model from Electra to Bigbird allows to have less iteration than before since Bigbird's maximum input length is longer than Electra.

