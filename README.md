# News Analyzer


## About
The **News Research Tool** is an AI application through which users can insert news articles, and ask the application questions about the article. This AI tool makes use of OpenAI's ChatGPT 3.5 Turbo as well as *LangChain, NLTK,* and *FAISS*. Since ChatGPT 3.5 does not have access to the internet and has only been trained on real world data up until January 2022, this application makes use of the other tools mentioned in order to read through the data in the news article and return a coherent response. You can read more about the underlying tools in the architecture section of this document. 
***
## Visit the Application Page
Here is a live version of the [News Research Tool]('https://mukundsnewsanalyzer.streamlit.app/') that was launched through the Streamlit service. 
***

## How to use the Application
![App Page Screenshot](/images/empty_res_page.png)
This is what the home landing page of the Application looks like. Use the following steps to have a pleseant experience:
1. Add URLs of News Articles or Online Documents that are not behind a paywall on the left sidebar. You can add up to 5 URLs.
2. Ask a question regarding any of the Articles you uploaded in the **Question** box. 
3. Click the **Enter** button

Example:
![App Page Screenshot](/images/news_res_ex.png)
In the above example, the following CNBC article [link](https://www.cnbc.com/2024/05/09/mcdonalds-makes-changes-to-increase-mobile-sales.html) about McDonlad's was provided as a sample article. Here is a paragraph from the article:
![McDonlads Article Screenshot](/images/McDonaldsArticle.png)
As seen above, the article credits a memo to McDonald's U.S. Customer Experience Officer, Tariq Hassan.
Therefore, the application was correct in its outputted answer.  

## Why Not Use ChatGPT?
1. People who don't have an account with OpenAI can use this service
2. ChatGPT 4 can perform the same functionality as the News Research Tool when given an article. However, this Application was built with ChatGPT 3.5 which does not have access to the internet and could only answer questions based on training data that was kept up to date until January 2022. 
3. Users of ChatGPT 3.5 can still copy and paste text into ChatGPT's prompt and ask it question based on the provided text, but it can be tedious to do so for many blocks of text. This Application provides a seamless input process.
4. ChatGPT 3.5 has a word limit of 3,000 words, and 4.0 has a limit of 25,000 words. 

## How Does it Work?!?
### A Look under the Hood
For a detailed explanation of the Application's architecture and code, I would highly recommend taking a look at the original [project tuorial]("https://www.youtube.com/watch?v=MoqgmWV1fm8") on YouTube by [Dhaval Patel]("https://www.linkedin.com/in/dhavalsays"). However, I will cover an overview of the architecture in the following sections. There are three main steps involved in the successful functioning of this Application:
1. Loading the Data
2. Splitting the Data
3. Embedding the Data

![Architecture Screenshot](/images/news_res_architecture.png)

The above image was taken from [Dhaval Patel's]("https://www.linkedin.com/in/dhavalsays") project [tuorial]("https://www.youtube.com/watch?v=MoqgmWV1fm8") at 11 minutes and 2 seconds.



### Loading the Data
The first step is to load data from online articles. Since GPT 3.5 does not have access to the internet, we will need to use a LangChain method called **UnstructuredURLLoader** to gather data from a link and retrieve it in the form of a dictionary. The documentation for the UnstructuredURLLoader can be found [here]("https://python.langchain.com/v0.1/docs/integrations/document_loaders/url/").


### Splitting the Data
The next step is to split our data. Since the OpenAI API charges a small fee for every 1000 input tokens an output tokens, it would be ideal to make our input text as small as possible while retaining the necessary information to get a strong result. This can be achieved with the help of splitting our data into chunks by using something called the **RecusriveTextSplitter**.

### Embedding the Data
In order to feed only the relevant chunks of our data to the OpenAI API, we will need to perform a process called **Embedding**. This process uses the FAISS (Facebook AI Similarity Search) [library]('https://github.com/facebookresearch/faiss/wiki) to seek out chunks of data that has information relevant to our question. This way, only these chunks are passed onto the OpenAI API along with the question. This greatly reduces the cost of using the OpenAI API. 


## Examples
Let us look at some examples of successful Sign Ups. 

***

## Sources and Origins 
As mentioned before, I learnt how to build this application through [Dhaval Patel's]("https://www.linkedin.com/in/dhavalsays") project [tuorial]("https://www.youtube.com/watch?v=MoqgmWV1fm8"). However, I've made some of my own changes to the functionality of the Application as well as the python code. 
1. I changed the placement of the **Enter**  button. While this may not seem like a big deal, there was an error that would pop up if a user did not hit the 'Process URLs' button from Dhaval's tutorial once again, even after submitting their question. In order to avoid this error, I changed some of the python code to be able to process the URLs and input the question in the click of a single button.
2. Since LangChain recently updated a lot of their software, many of the methods and functions that Dhaval uses in his tutorial are outdated. The most notable of these, which many other developers got stuck on, is the following python code. 
 ```python
   # storing results of the vector index 
    file_path = "vector_index.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_openai, f)

    #opening the file
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
```
After LangChain updated their software, the ```vectorIndex``` is stored as an object which can't be parsed. Therefore, one has to serialize it into bytes before writing it into a file and then desearilize it after reading it from a file. I had to do a lot of research and digging around to be able to create a solution that worked:
```python
# storing results of the vector index 
file_path = "vector_index.pkl"
with open(file_path, "wb") as f:
    pickle.dump(vectorindex_openai.serialize_to_bytes(), f)

#opening the file
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        uploaded_pickle = pickle.load(f)
        vectorIndex = vectorindex_openai.deserialize_from_bytes(serialized=uploaded_pickle, embeddings=embeddings)
```
3. Apart from these two changes, I made minor changes like changing the title and adding a paragraph on the bottom left to credit Dhaval Patel and his tutorial.
***

## Want to Create Your Own News Analyzer?
### Build From Scratch
If you would like to build this Application from scratch, you can watch [Dhaval Patel's]("https://www.linkedin.com/in/dhavalsays") project [tuorial]("https://www.youtube.com/watch?v=MoqgmWV1fm8") on YouTube. 

### OR

### Download and customize from my uploaded code
Make sure to check the *dependencies.txt* file and the *requirements.txt* file to know what dependencies this project needs. 

Note: You will also need to set up you own OpenAI API key. Never openly type this key into your code or upload it anywhere online. Make sure to store it as an environmental variable in a .env file and include the .env file in your .gitignore file.

If the above paragraph went completely over your head, then it's worth it to check out [Dhaval Patel's]("https://www.linkedin.com/in/dhavalsays") project [tuorial]("https://www.youtube.com/watch?v=MoqgmWV1fm8") where he walks through everything step by step.

If you wish to deploy your app through Streamlit like I did, you can add your OpenAI API in their Secrets file that you can access by clicking *Advanced Settings* as shown below.

![Advanced Settings](/images/deployST.png)

![Advanced Settings](/images/STsecrets.png)

Here is a [tutorial]('https://www.youtube.com/watch?v=oWxAZoyyzCc') by Misra Turp that walks you through how to add your OpenAI API key in Streamlit. 

## All the Best To You!
