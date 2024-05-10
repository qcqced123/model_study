import os, sys
import openai
import google.generativeai as genai
import pandas as pd

from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from preprocessing import cleaning_words

load_dotenv()


def image2doc(image_path: str) -> str:
    """ Convert image to text document by using Langchain Community
    """
    image_loader = UnstructuredImageLoader(image_path)
    output = image_loader.load()[0]
    return output


def pdf2doc(pdf_path: str) -> str:
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
        https://arxiv.org/abs/2005.14165
        https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        https://platform.openai.com/docs/guides/text-generation
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


def google_gemini_api(paper_link: str, foundation_model: str = 'gemini-pro', temperature: float = 0) -> str:
    """ make Arxiv Questioning & Answering dataset function with Google AI Gemini API

    As you run this function before, you must set up the your own Google API key for the Gemini API.
    you can use the gemini-pro-api for free with the Google API key.

    we will use the Zero-Shot Learning for generating the QA dataset from the given paper link.

    Args:
        paper_link (str): the paper link for QA dataset
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
    # context = pdf2doc(paper_link)
    datasets = ''
    try:
        prompt = f"[context] {paper_link}"\
                 f"Please make some questions and answers from [context]." \
                 f"When creating your questions and answers from [context]," \
                 f"please generate much longer and detailed information into questions and answers" \
                 f"And also, make output shape as below: " \
                 f"Questions: 1., 2., 3. ... 10., Answers: 1., 2., 3. ... 10." \
                 f"Don't use bold, italic, or underline text in the questions and answers."

        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config,
        )
        datasets = response.text

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


def remove_garbage(text: str) -> str:
    """ remove garbage text from arxiv paper
    """
    text = text[text.find("v i X r a\n\n"):]
    return text


def build_qa_dataset(text: str) -> pd.DataFrame:
    """ build QA dataset from the given paper link

    Args:
        text (str): the text (Question and Answering from Open Ai GPT-3.5 API or Google AI Gemini API)
    """
    questions_text = text.split("Questions:")[1].split("Answers:")[0].strip()
    answers_text = text.split("Questions:")[1].split("Answers:")[1].strip()

    questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
    answers = [a.strip() for a in answers_text.split("\n") if a.strip()]
    df = pd.DataFrame({"Questions": questions, "Answers": answers})
    return df


if __name__ == '__main__':
    link = 'p_tuning.pdf'
    test = pdf2doc(f'./data_folder/arxiv_qa/papers/{link}')

    test = remove_garbage(test)
    clean_text = cleaning_words(test)  # remove all of trash text such as this papers pid
    print(test)

    text = google_gemini_api(clean_text)
    print(text)

    df = build_qa_dataset(text)
    print(df)

    df.to_csv('p_tuning.csv', index=False)




