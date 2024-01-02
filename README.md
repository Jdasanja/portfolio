# portfolio
AI IN THE WORKPLACE: ANALYZING EMPLOYEE SENTIMENTS
Jonathan Asanjarani
Department of Data Analysis and Visualization, CUNY Graduate Center
LLM Fall 2023: Large Language Models and ChatGPT
Professor Michelle McSweeney
December 18, 2023
 
Abstract
The purpose of this project is to apply BERT to classify employee sentiments towards their company. This project will use natural language processing (NLP) and sentiment analysis. Specifically, it will employ BERT, a language model hosted by Hugging Face, to perform a simple sentiment classification task. This project will provide samples of employee reviews to the BERT model to see how accurately it predicts employee sentiments. A sentiment can either be classified as “Positive” or “Negative”. The fine-tuning process of the BERT model will be carried out using TensorFlow, an open-source machine learning platform equipped with needed deep learning libraries. To train the model, a dataset of employee reviews sourced from Kaggle will be used. The dataset, accessible at https://www.kaggle.com/datasets/fireball684/hackerearthericsson/, consists of job reviews gathered from Glassdoor. 
The focus of this lab is to give an overview of the setup and tuning procedures of the BERT model when applied to analyze employee sentiment in reviews. Furthermore, we will explore the societal and ethical considerations tied to utilizing language models for monitoring sentiments in the workplace.












Background
BERT
This project employs BERT to analyze employee sentiments. BERT is a language model developed in 2018, which utilizes neural networks to represent text. It is designed to understand the context and meaning of words in a sentence by looking at both the left and right words around them. This enables BERT to have a more accurate representation of complex linguistic relationships. The architecture for BERT was made possible after developers began implementing an “Attention Mechanism” into models in 2015.
An attention mechanism is the ability for a model to embed “weighted” values for each word. These “weighted” values give a relevant context for each unit. The “Transformer” model was created based on this “attention mechanism.”  
A Transformer is a model with a decoding and encoding component. The encoder encodes the input sentence and passes it to the decoder, where a representation is decoded for a relevant task. The encoder contains two layers. Encoded texts are first encoded through a “self-attention” layer, which identifies relevant parts of a word and encodes a center word in the input sentence. The output of this layer is than fed through a feedforward neural network. Each output is passed through the feedforward neural network independently. The decoder also has a self-attention and feedforward layer, but it is accompanied by an “attention” layer, which helps the decoder find relevant parts of a sentence.
There are multiple variations of pre-trained transformer models. Some models only use decoders or encoders. BERT is an encoder-only pre-trained transformer model. BERT stands for Bidirectional Encoder Representations from Transformers. The name implies that it can develop a more accurate representation of contextual relationships by using bidirectional training.
Bidirectional training uses a “masked language model” (MLM) pre-training objective. This masked language model randomly masks tokens (words converted into individual units) at random, from the input. The objective is for the model to predict the original vocabulary of the masked word based only on its context. 
The innovative aspect of BERT is that it can be fine-tuned to perform specific tasks. It would only require adding a task specific layer to a pre-trained BERT model. BERT can be modified to perform specific tasks without having to rebuild the entire model.
Sentiment Analysis
Sentiment Analysis has been extensively researched in the context of natural language processing (NLP). The goal of Sentiment Analysis is to systematically study people’s opinions, sentiments, and emotions using computational methods. Large Language Models perform well on various sentiment analysis tasks. In a study conducted by Stanford University, 13 sentiment analysis tasks were studied across 26 datasets to determine how well LLMs could comprehend subjective information. The results indicated that Large language models (LLMs) do well without specific training in simple sentiment analysis tasks, like classifying sentiment as positive or negative. However, for more complicated tasks that require a deep understanding of specific sentiments or structured sentiment information, LLMs are not as good as traditional models trained with specific, in-domain data.






Social & Ethical Implications


Social Implications
According to Leslie Hammer, a professor at Oregon Health & Science University and co-director of the Oregon Healthy Workforce Center, “When employees feel they are not cared for or trusted by their employers, they are likely to have lower levels of commitment to the organization and perceive lower levels of psychological safety and higher levels of stress, all negatively affecting the relationship between employees and their employers, and specifically their managers and supervisors.” 45% of employees that work at companies who use AI to monitor their workforce report negative mental health effects. Data shows that 32% of employees who are monitored with tech during the workday by their employer report their mental health as poor or fair (as opposed to good or excellent) compared with 24% who are not monitored.  This is likely because most Americans see a greater chance for potential downsides than upsides when it comes to AI. Most workers feel they would “definitely” be inappropriately watched. A majority (66%) also agree that the information collected would be misused. However, there are some potential upsides. Roughly 49% of Americans believe workplace security would improve. 46% of Americans believe inappropriate behavior would decrease. A longitudinal study conducted in Germany examined the impact AI had on workers life and jobs satisfaction. Workers overall reported feeling less satisfied with their job and life and more concerned with their job security and personal financial situation.
Ethical Implications
Large Language Models and Other AI tools can be used for workplace surveillance and employee monitoring. Employees are constantly producing data which can be used to analyze performance, well-being, and behavior.  Some companies only monitor how much time employees spend on different apps while other companies go as far as remotely viewing employees’ screens and capturing videos of their cameras. Aside from the ethical risk of this form of micromanaging, there is also a risk of AI being trained on private or unethically sourced data. In 2020, The Information Commissioner’s Office (ICO) investigated Barclays Bank, over their use of software that let managers monitor how much time an employee spent away from their desk. The bank monitored their employees and were monitoring their staff anonymously for 18 months but eventually began tracking individual employees. Barclays was fined over 1.1 billion dollars for this transgression.
Since the pandemic, there has been an increasing demand for workplace monitoring solutions while appropriate regulations have yet to arrive. More recently, Harding Published an article in 2021 for the Mozilla foundation, revealing that employers can potentially have access to private chat messages from apps like Slack, Microsoft Teams, or Zoom. Employers need to submit a request to Slack to gain access to your personal conversations. Your employer also needs to prove that you either gave them permission or have a valid legal reason for accessing your personal conversations. Microsoft Teams will work with your company’s admin team to provide extensive data metrics like how many messages a user sends, how many audio and video calls they’re engaging in, and how often they’re setting up meetings and more. While Zoom does not allow users to enter a meeting without being visible, it does allow admins to listen to calls without both parties being aware.
The ethical risks of AI in workplaces are evident in cases like Barclays Bank's fine of over $1.1 billion for unauthorized employee monitoring. The increased demand for workplace monitoring solutions, as well as the potential privacy breaches employers are at risk for, underscores the need for clear regulations to balance employer surveillance and employee privacy. As AI technologies advance, establishing ethical guidelines becomes crucial to protect individuals' privacy rights at work.


Method
In this test, the BERT model was fine-tuned with a dataset of employee reviews sourced from Kaggle. The dataset, accessible at https://www.kaggle.com/datasets/fireball684/hackerearthericsson/, consisted of job reviews gathered from Glassdoor. The model was then presented with a separate series of  reviews from the site, https://www.careerbliss.com, to evaluate whether the language model would reflect the same sentiments expressed by the employees. 
BERT is a versatile model that can be used for a myriad of NLP (Natural Language Processing) tasks, including sentiment analysis. BERT is accessed through Hugging face, a repository that hosts machine learning models. Users can access, share, and collaborate on various machine learning models, including BERT.
This lab applied Tensorflow as the underlying framework used for implementing machine learning models. Like Pytorch, Tensorflow are open-source machine learning libraries that focus particularly on deep neural networks. Additionally, since BERT is a pre-trained Transformer model, the Transformers library, developed by Hugging Face, is used.
Using the Pandas library, I loaded the employee review data set into a tabular data structure. To make the sentiment labels of “positive” and “negative” compatible with machine learning models, a function was used to convert these categorical values into binary values. The function was then applied to the dataset, which converted the “positive” and “negative” labels into “0” and “1”.
Following the data cleaning process, the BERT Tokenizer is used to tokenize the words in the dataset. Tokenization is the process of breaking down texts into individual units. Once the individual units are tokenized, the input is converted into an ‘Input Example’ class. ‘Input Example’ is a class provided by the Transformers library. It represents a single training example and should consist of all the input data.
To feed the dataset into the BERT model, the input data needs to be converted into a Tensorflow dataset. This can be achieved using the ‘convert_examples_to_tf_dataset’ function. This will tokenize and format the data into readable form for the BERT model.
Before the model is trained, it needs to be compiled with essential components in order to maximize performance. For this lab, I utilized an optimizer called “Adam” to enhance its ability to make predictions. The “sparse categorical crossentropy” is used to evaluate how well the model does when it predicts a review. Furthermore, we evaluate the percentage of correct predictions using the ‘accuracy’ measure.
We can now train the model on the tokenized and newly formatted data using the ‘fit’ function. This project specifies that the model be trained for two epochs. Once the model is finished training, users can provide a sample sentence for sentiment prediction. The BERT tokenizer is used to tokenize the sample and the trained model is employed to make a prediction.
The github can be found here: https://github.com/Jdasanja/LLMFAL2023/blob/main/BERT_Lab_updated.ipynb

Results
	During the evaluation phase, the employee sentiment analyzer was tested on a sample of 10 employee reviews. The results revealed that the sentiment predictions made by the employee sentiment analyzer aligned with the sentiments expressed by the employees in each case. Specifically, the model accurately classified the sentiment in all 10 instances, demonstrating its proficiency in discerning and categorizing sentiments within the context of employee reviews.





Conclusion
BERT successfully demonstrated its ability to effectively classify binary sentiment relationships. During the evaluation phase, which tested the employee sentiment analyzer on a sample of 10 employee reviews, the model accurately classified employee sentiments in all instances. This highlights its ability to accurately classify basic employee sentiments. 
Limitations of Existing Research
The accuracy of positive and negative sentiment analysis shows insight into the overall performance of using the BERT model to analyze sentiment. However, there are obvious limitations to this binary classification approach. Positive and Negative sentiments only encompass a fraction of the range of human emotion. Due to this approach, it is impossible to study nuanced or subtle employee reviews that may not fit into either sentiment.
Directions for Future Research
Nuanced or subtle employee messages require a broader spectrum of sentiments to be incorporated in the training data. Including neutral or mixed emotions in the training data could provide a more comprehensive review of employee feedback and could also provide a more wholistic understanding of how well BERT classifies employee sentiments. When contextualizing the broader landscape of sentiment analysis within natural language processing, it is important to note that Large Language models have difficulty attaining a deep understanding of complex and specific emotions. This type of sentiment classification task typically requires more structured sentiment information with traditional models trained in domain specific data. These observations underline the importance of ongoing research to expand the capabilities of models like BERT to capture a richer understanding of human sentiment.
	
 
References
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019, May 24). Bert: Pre-training of deep bidirectional Transformers for language understanding. arXiv.org. https://arxiv.org/abs/1810.04805 
Lerner, M. (2023, September 7). Electronically monitoring your employees? it’s impacting their mental health. American Psychological Association. https://www.apa.org/topics/healthy-workplaces/employee-electronic-monitoring 
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023, August 2). Attention is all you need. arXiv.org. https://arxiv.org/abs/1706.03762 
Weidinger, L., Gabriel, I., Irving, G., Legassick, S., Isaac, W., Hendricks, L. A., Rimell, L., Haas, J., Birhane, A., Biles, C., Stepleton, T., Hawkins, W., Brown, S., Kenton, Z., Kasirzadeh, A., Balle, B., Glaese, C.-M., Myra, Huang, P.-S., … Mellor, J. (2021, December 8). Ethical and social risks of harm from language models - arxiv.org. https://arxiv.org/pdf/2112.04359.pdf 
Giuntella, O. & K. (2023, February 2). Artificial Intelligence and Workers’ well-being. IZA Discussion Papers. https://ideas.repec.org/p/iza/izadps/dp16485.html 


