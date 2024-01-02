# AI IN THE WORKPLACE: ANALYZING EMPLOYEE SENTIMENTS

**Jonathan Asanjarani**  
*Department of Data Analysis and Visualization, CUNY Graduate Center*  
*LLM Fall 2023: Large Language Models and ChatGPT*  
*Professor Michelle McSweeney*  
*December 18, 2023*

## Abstract

This project aims to employ BERT for classifying employee sentiments toward their company, utilizing natural language processing (NLP) and sentiment analysis. BERT, a language model hosted by Hugging Face, will be applied to a sentiment classification task using TensorFlow for fine-tuning. The dataset, sourced from Kaggle [here](https://www.kaggle.com/datasets/fireball684/hackerearthericsson/), comprises job reviews from Glassdoor. The focus is on providing an overview of BERT model setup and tuning for employee sentiment analysis, along with exploring societal and ethical considerations tied to language model use in workplace sentiment monitoring.

## Background

### BERT

This project leverages BERT, a language model developed in 2018, utilizing neural networks for text representation. BERT understands word context by examining surrounding words on both sides, providing a more accurate representation of linguistic relationships. The architecture incorporates an “Attention Mechanism,” embedding weighted values for each word. The “Transformer” model, based on this mechanism, consists of an encoder with two layers. BERT, being an encoder-only model, stands for Bidirectional Encoder Representations from Transformers. It employs bidirectional training using a “masked language model” (MLM) pre-training objective. This allows BERT to develop accurate contextual relationships without rebuilding the entire model, making it versatile for specific tasks.

### Sentiment Analysis

Sentiment analysis studies opinions, sentiments, and emotions using computational methods in the context of NLP. Large Language Models (LLMs) perform well in simple sentiment analysis tasks, classifying sentiment as positive or negative. However, for more complex tasks requiring a deep understanding of specific sentiments, LLMs may not outperform traditional models trained with specific, in-domain data.

## Social & Ethical Implications

### Social Implications

Leslie Hammer states that employees feeling uncared for or untrusted by employers can lead to lower commitment, perceived psychological safety, and higher stress levels. Companies using AI for workforce monitoring report negative mental health effects among employees. There is skepticism and concern among workers regarding potential misuse of collected information. While some anticipate improved workplace security, others express worries about inappropriate behavior and misuse.

### Ethical Implications

Large Language Models and AI tools in workplace surveillance raise ethical concerns. Companies may monitor employees extensively, posing risks of micromanaging and potential AI training on private or unethically sourced data. Cases like Barclays Bank's fine for unauthorized employee monitoring underscore the need for clear regulations balancing surveillance and privacy. The increased demand for workplace monitoring solutions during the pandemic, without adequate regulations, emphasizes the importance of ethical guidelines as AI technologies advance.

## Method

The BERT model was fine-tuned using a Kaggle dataset of employee reviews. The dataset, accessible [here](https://www.kaggle.com/datasets/fireball684/hackerearthericsson/), consists of job reviews from Glassdoor. TensorFlow, an open-source machine learning platform, was employed for fine-tuning. The BERT Tokenizer was used for text tokenization, and the dataset was converted into a TensorFlow dataset. The model was compiled with an "Adam" optimizer for prediction enhancement and "sparse categorical crossentropy" for evaluation. Training involved two epochs, and the model was tested on a separate set of reviews from [CareerBliss](https://www.careerbliss.com).

The code for this project is available on [GitHub](https://github.com/Jdasanja/LLMFAL2023/blob/main/BERT_Lab_updated.ipynb).

## Results

During the evaluation phase, the employee sentiment analyzer accurately predicted sentiments in a sample of 10 employee reviews. The model demonstrated proficiency in discerning and categorizing sentiments, aligning with employees' expressions in all instances.

## Conclusion

BERT effectively classified binary sentiment relationships, accurately categorizing basic employee sentiments. However, limitations exist in this binary classification approach, as it excludes nuanced or subtle employee reviews. Future research should incorporate a broader spectrum of sentiments, including neutral or mixed emotions, to provide a more comprehensive understanding of employee feedback and enhance sentiment classification.

## References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019, May 24). [Bert: Pre-training of deep bidirectional Transformers for language understanding](https://arxiv.org/abs/1810.04805).
- Lerner, M. (2023, September 7). [Electronically monitoring your employees? it’s impacting their mental health](https://www.apa.org/topics/healthy-workplaces/employee-electronic-monitoring).
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023, August 2). [Attention is all you need](https://arxiv.org/abs/1706.03762).
- Weidinger, L., Gabriel, I., Irving, G., Legassick, S., Isaac, W., Hendricks, L. A., Rimell, L., Haas, J., Birhane, A., Biles, C., Stepleton, T., Hawkins, W., Brown, S., Kenton, Z., Kasirzadeh, A., Balle, B., Glaese, C.-M., Myra, Huang, P.-S., … Mellor, J. (2021, December 8). [Ethical and social risks of harm from language models - arxiv.org](https://arxiv.org/pdf/2112.04359.pdf).
- Giuntella, O. & K. (2023, February 2). [Artificial Intelligence and Workers’ well-being](https://ideas.repec.org/p/iza/izadps/dp16485.html).

