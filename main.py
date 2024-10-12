import fitz 
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import dspy

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
# Set TOKENIZERS_PARALLELISM to false to avoid parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
FOLDER_PATH = 'data'

def extract_text_from_pdf(pdf_path):
    texts_with_info = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            texts_with_info.append((text, page_number, pdf_path))
    return texts_with_info

def chunk_text(text, chunk_size=10000):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"---->>>>Chunked text: {chunks}")  # Debugging print statement
    return chunks

# Configure the language model (LM)
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)

# Define a custom rationale
rationale_type = dspy.OutputField(
    prefix="Reasoning: Let's think step by step in order to",
    desc="${produce the answer}. We ..."
)

# Create a signature for the ChainOfThought module
class QAWithRationale(dspy.Signature):
    question = dspy.InputField()
    context = dspy.InputField()
    rationale = rationale_type
    answer = dspy.OutputField(desc="often around 500000 words without counting with citation page number and file name")

# Initialize ChainOfThought with the custom signature
generate_answer = dspy.ChainOfThought(QAWithRationale)

model = SentenceTransformer('all-MiniLM-L6-v2')

def ask_openai_question(text_chunk, question, page_number, filename):
    print(f"---->>>>Question asked: {question} {page_number} {text_chunk}")
    response = generate_answer(question=question, context=text_chunk)
    rationale, answer = response.rationale, response.answer
    print(f"---->>>>Rationale: {response.rationale}")
    return rationale, answer, page_number, filename

def evaluate_and_combine_answers(answers):
    # Improved logic to evaluate and combine answers
    if not answers:
        return "No related data found."
    # Using language model to refine comprehensive answer
    combined_text = " ".join(answers)
    response = generate_answer(question="You are not creative. You are an analytical NGO auditor. Analize all the answers from all pages and files, select the most strong answers from each page and file relate to the question, combine all answers in one detailed answer with examples, actions, strategies, intitiatives and project names, always cite the page number and file name! Be extremely detailed, avoid be prolixo.", context=combined_text)
    return response.answer

def is_relevant_answer(answer, context):
    context_embedding = model.encode(context, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(context_embedding, answer_embedding)
    threshold = 0.5
    return similarity.item() > threshold

def main():
    print("---->>>>Iterating over PDF files in folder...")
    all_answers = []
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith('.pdf'):
            file_path = os.path.join(FOLDER_PATH, filename)
            print(f"---->>>>Extracting text from {filename}...")
            texts_with_info = extract_text_from_pdf(file_path)

            question = "How save the children align to climate projects?"
            for text_chunk, page_number, filename in texts_with_info:
                rationale, answer, page_number, filename = ask_openai_question(text_chunk, question, page_number, filename)
                if is_relevant_answer(answer, text_chunk):
                    citation = f"(page {page_number} from file {filename})"
                    all_answers.append(f"{answer} {citation}")
                    print(f"---->>>>all_answers: {citation} {answer}")
                else:
                    print("-----XXX----No related information found in this chunk")


    print("---->>>>Evaluating and combining answers...")
    comprehensive_answer = evaluate_and_combine_answers(all_answers)
    citation = "" if not all_answers else ""
    print(f"---->>>>Comprehensive Answer: {comprehensive_answer} {citation}")

if __name__ == "__main__":
    main()
