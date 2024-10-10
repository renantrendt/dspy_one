import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer, util
import dspy

# Load environment variables
OPENAI_API_KEY = 'sk-proj-***'
FOLDER_PATH = 'data'

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Configure the language model (LM)
lm = dspy.LM('openai/gpt-4o-mini', api_key=OPENAI_API_KEY)
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
    answer = dspy.OutputField(desc="often between 1 and 10 words")

# Initialize ChainOfThought with the custom signature
generate_answer = dspy.ChainOfThought(QAWithRationale)

model = SentenceTransformer('all-MiniLM-L6-v2')

def ask_openai_question(text, question):
    response = generate_answer(question=question, context=text)
    return response.rationale, response.answer

def evaluate_and_combine_answers(answers):
    # Improved logic to evaluate and combine answers
    if not answers:
        return "No answers provided."
    # Using language model to refine comprehensive answer
    combined_text = " ".join(answers)
    response = generate_answer(question="Create a cohesive summary of the following information.", context=combined_text)
    return response.answer

def is_relevant_answer(answer, context):
    context_embedding = model.encode(context, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(context_embedding, answer_embedding)
    threshold = 0.0
    return similarity.item() > threshold

def main():
    print("---->>>>Iterating over PDF files in folder...")
    all_answers = []
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith('.pdf'):
            file_path = os.path.join(FOLDER_PATH, filename)
            print(f"---->>>>Extracting text from {filename}...")
            text = extract_text_from_pdf(file_path)
            question = "How save the children align to climate projects?"
            print(f"---->>>>Asking question: {question} for {filename}")
            print(f"---->>>>Content: {text}")
            rationale, answer = ask_openai_question(text, question)
            print(f"---->>>>Rationale: {rationale}")
            print(f"---->>>>Answer: {answer}")
            if is_relevant_answer(answer, text):
                all_answers.append(answer)
            else:
                print("No related information found.")

    print("Evaluating and combining answers...")
    comprehensive_answer = evaluate_and_combine_answers(all_answers)
    print(f"---->>>>Comprehensive Answer: {comprehensive_answer}")

if __name__ == "__main__":
    main()
