# Introduction to NLP - Assignment 3 (Sequence to Sequence)

### Name: Gokul Vamsi Thota
### Roll number: 2019111009

## Model checkpoints

* Language model trained on europarl corpus (english sentences) can be found [here](https://drive.google.com/file/d/1IMSYmLnPEvRFLSrFJ_Ahl9EqGyvJPxhD/view?usp=sharing). 

* Language model trained on news crawl corpus (french sentences) can be found [here](https://drive.google.com/file/d/1ycA4TfuvsnnWagQOZl-4PQDKZH-vXI1_/view?usp=sharing).

* Machine translation model trained on ted talks corpus without fine tuning can be found [here](https://drive.google.com/file/d/1QVq1d018Gf-91V0BR539LQzv2-A9bpec/view?usp=sharing).

* Machine translation model trained on ted talks corpus with fine tuning can be found [here](https://drive.google.com/file/d/1n9poC_YIRsTLVgwU114tr16AH0waK3OT/view?usp=sharing).

## Instructions to execute

* All the results obtained: text files for language model - perplexity scores, bleu scores; for all cases are in the `results` folder.

* It is assumed that the 3 corpus folders are present in the same folder in which 2019111009_assignment3 folder is present, before executing

### Prompt

* Inside the folder 2019111009_assignment3 (folder containing language_model.py), execute the language model with the command: `python3 language_model.py <path_to_model>`. Ensure that the path is valid and contains the correct model.

* Inside the folder 2019111009_assignment3 (folder containing machine_translation.py), execute the machine translation model with the command: `python3 machine_translation.py <path_to_model>`. Ensure that the path is valid and contains the correct model.

* It is assumed that only an english sentence is entered on the prompt, for both the tasks.

In both of above cases, after executing the command, a prompt is shown asking for 'input sentence: ', enter the required sentence, and the corresponding probability of that sentence occurring (in the first case) or the french translation (in the second case) is displayed. 

After printing this value, this prompt is shown again, and the same process can continue. Press `CTRL + C` or enter `-1` as sentence to terminate the execution process and to exit the prompt.

### Calculation of perplexity and bleu

* For the language model, uncomment lines 145 and 146 where perplexity values are generated and stored in required format.

* For the translation, uncomment line 188 for storing bleu scores for all sentences / corpus bleu score, stored in required format.

## Additional Information / Assumptions

* Maximum length (max number of words per sentence) was chosen based on number of words per sentences, based on the task, and appropriate padding was done. For the first task, values of 11 and 8 were chosen for english and french; while 6 was used for both english and french in second task.

* Vocabulary size of 5000 was used for the language modelling task and 8000 was used for the translation task (arrived at after experimenting).