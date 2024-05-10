import os, sys
import openai
import google.generativeai as genai
import pandas as pd

from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader


load_dotenv()


def image2doc(image_path: str) -> str:
    """ Convert image to text document by using Langchain Community
    """
    image_loader = UnstructuredImageLoader(image_path)
    output = image_loader.load()[0]
    return output


def pdf2doc(pdf_path: str) -> Document:
    """ Convert pdf to text document by using Langchain Community
    """
    pdf_loader = UnstructuredPDFLoader(file_path=pdf_path)
    output = pdf_loader.load()[0].page_content
    return output


def openai_gpt3_api(script: str, foundation_model: str = 'gpt-3.5-turbo-16k', temperature: float = 0) -> List:
    """ extract food ingredients from the given text (YouTube video script) with GPT-3.5 Inference API (OpenAi)

    Args:
        script (str): The string text from YouTube video for extracting food ingredients
        foundation_model (str): The foundation model for extracting food ingredients from the given text,
                                default is 'gpt-3.5-turbo-16k'
        temperature (float): default 0.0, the temperature value for the diversity of the output text
                             (if you set T < 1.0, the output text will be more deterministic, sharpening softmax dist)
                             (if you set T > 1.0, the output text will be more diverse, flattening softmax dist)
    Returns:
        ingredient_data: list of food ingredients from the given text

    References:

    """
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    openai.api_key = OPENAI_API_KEY

    ingredient_data = []
    try:
        prompt = "Please provide a 'Food Ingredient List' in a 'structured format'." \
                 "The format should include only the 'ingredient name' without quantities or units." \
                 "If you don't have an ingredient list, simply write 'No ingredient'." \
                 "When compiling the 'Food Ingredient List' from the given [script], ensure that it is written in 'English'." \
                 "Additionally, exclude non-food items and focus soely on 'edible ingredients'." \
                 "For example: - Egg, - Cheese, - Tomato, - Salt, - Sugar"
        response = openai.ChatCompletion.create(
            model=foundation_model,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': script},
            ],
            temperature=temperature
        )

    except Exception as e:
        print(e)

    return ingredient_data


def google_gemini_api(paper_link: str, example: str, foundation_model: str = 'gemini-pro', temperature: float = 1) -> List:
    """ make Arxiv Questioning & Answering dataset function with Google AI Gemini API

    As you run this function before, you must set up the your own Google API key for the Gemini API.
    you can use the gemini-pro-api for free with the Google API key.

    Args:
        paper_link (str): the paper link for QA dataset
        example (str): the example dataset for QA dataset and 1-shot learning
        foundation_model (str): The foundation model for extracting food ingredients from the given text,
                                default is 'gemini-pro'
        temperature (float): default 0.0, the temperature value for the diversity of the output text
                             (if you set T < 1.0, the output text will be more deterministic, sharpening softmax dist)
                             (if you set T > 1.0, the output text will be more diverse, flattening softmax dist)

    References:
        https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/tutorials/quickstart_colab.ipynb?hl=ko#scrollTo=HTiaTu6O1LRC
        https://ai.google.dev/gemini-api/docs/get-started/python?hl=ko&_gl=1*7ufqxk*_up*MQ..*_ga*MTk2ODk3NDQyNi4xNzE0OTIwMjcw*_ga_P1DBVKWT6V*MTcxNDkyMDI2OS4xLjAuMTcxNDkyMDI2OS4wLjAuOTQwNDMwMTE.
        https://ai.google.dev/gemini-api/docs/quickstart?hl=ko&_gl=1*12k4ofq*_up*MQ..*_ga*MTk2ODk3NDQyNi4xNzE0OTIwMjcw*_ga_P1DBVKWT6V*MTcxNDkyMDI2OS4xLjAuMTcxNDkyMDI2OS4wLjAuOTQwNDMwMTE.
        https://ai.google.dev/api/python/google/generativeai/GenerativeModel?_gl=1*1ajz3qu*_up*MQ..*_ga*MTk2ODk3NDQyNi4xNzE0OTIwMjcw*_ga_P1DBVKWT6V*MTcxNDkyNDAyOC4yLjAuMTcxNDkyNDAyOC4wLjAuMTkwOTQyMjU0#generate_content
    """
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(foundation_model)
    generation_config = genai.types.GenerationConfig(
        candidate_count=1,
        temperature=temperature
    )
    context = pdf2doc(paper_link)
    datasets = []
    try:
        prompt = f"[context] {context}"\
                 f"[example] {example}." \
                 f"Please make some questions and answers from [context]." \
                 f"When creating your questions and answers, use the format and intent of the [example] questions and answers as a guide."

        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config,
        )
        dataset = response.text
        print(dataset)

    except Exception as e:
        print(e)

    return datasets


def get_paper_from_list(base_path: str) -> None:
    """ get paper from the list of papers

    Args:
        base_path (str): the base path for the list of papers
    """
    df = pd.DataFrame(columns=['paper_id'])
    paper_list = os.listdir(base_path)

    df['paper_id'] = [".".join(pid.split('.')[0:2]) for pid in paper_list]
    df.to_csv('paper_id_list.csv', index=False)
    return


if __name__ == '__main__':
    link = './deberta.pdf'
    example = """Question 1. What is the central research question or hypothesis that this paper addresses? 
    Answer 1. Based on my reading, the central research question this paper addresses is: 
    Given an encoding or representation of an image produced by a model like SIFT, HOG, or a convolutional neural network (CNN), to what extent is it possible to reconstruct or invert the original image?
    The authors propose a general framework for inverting image representations by posing it as an optimization problem - finding an image that best matches the target representation while conforming to natural image priors.
    They apply this technique to study and visualize the information captured by different representations, especially the layers of deep CNNs trained on ImageNet.
    In summary, the main hypothesis is that by inverting image representations, they can gain insights into the invariances captured by the representation as well as understand what visual information is preserved in models like CNNs. 
    The reconstructions allow them to analyze and visualize the encoding learned by the models.
    """

    example2 = """
    Q1. What is the main challenge addressed in this paper regarding image representations?    
    A1. The paper tackles the challenge of understanding the information encoded within image representations, particularly those generated by deep Convolutional Neural Networks (CNNs). Traditionally, it's difficult to grasp what specific visual features these representations capture.
    
    Q2. How do the authors propose to analyze image representations?
    A2. The authors introduce a novel approach: inverting the image representation. They formulate it as an optimization problem. The goal is to find an image that best matches the target representation (produced by models like SIFT, HOG, or CNNs) while also adhering to natural image properties (like smoothness and texture). By reconstructing the image from its representation, they can analyze the encoded information.
    
    Q3. What types of image representations are investigated in the paper?
    A3. The paper explores inverting various image representations, including:
    Shallow representations: Techniques like SIFT (Scale-Invariant Feature Transform) and HOG (Histogram of Oriented Gradients).
    Deep representations: The focus is on inverting representations learned by deep CNNs trained on large datasets like ImageNet.
    
    Q4. What is the key finding regarding information encoded by deeper CNN layers?
    A4. The study reveals that as you go deeper within a CNN architecture, the reconstructed images exhibit increasing invariance to details. In simpler terms, deeper layers become less sensitive to minor variations in the original image, while still retaining the semantic content (like the object depicted).
    
    Q5. How do the authors analyze the information distribution within CNN neurons?
    A5. They achieve this by inverting representations derived from subsets of CNN neurons. This analysis sheds light on the locality and specialization of information encoded across different channels and layers within the network.
    
    Q6. What is the overall benefit of inverting image representations, particularly for deep CNNs?
    A6. Inverting these representations provides valuable insights into two key aspects:
    Invariances: It helps understand what kind of variations the model disregards when encoding an image.
    Information Content: It reveals what visual information (like shapes, textures, or colors) is captured and preserved within the CNN's encoding.
    By analyzing the reconstructions, researchers gain a deeper understanding of how these models process and represent visual information.
    """

    # google_gemini_api(link, example)
    get_paper_from_list('./data_folder/arxiv_qa/papers/')