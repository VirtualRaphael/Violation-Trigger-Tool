pip install pandas scikit-learn transformers torch tqdm openpyxl PyMuPDF

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import fitz
import re
import string

# Load the saved model and tokenizer
model_save_path = 'bert_model.pt'
tokenizer_save_path = 'bert_tokenizer'

tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(model_save_path))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def classify_text(text, model, tokenizer, device):
    encodings = tokenizer(text, truncation=True, padding=True, return_tensors='pt', max_length=512)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1)
    return prediction.item()

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    return [page.get_text("text") for page in document]

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_formatting(text):
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'Document Code:.*\n', '', text)
    text = re.sub(r'Copyright.*\n', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text

def extract_relevant_pages(pages_text):
    start_keywords = ["Introduction", "Purpose"]
    end_keywords = ["Appendices", "Appendix"]
    start_idx, end_idx = 0, len(pages_text)

    for i, text in enumerate(pages_text):
        if any(keyword in text for keyword in start_keywords):
            start_idx = i
            break
    for i, text in enumerate(pages_text):
        if any(keyword in text for keyword in end_keywords):
            end_idx = i
            break
    return pages_text[start_idx:end_idx]

def split_into_sections(text):
    sections = re.split(r'\n{2,}|\n(?=\w.+?:)', text)
    return [section.strip() for section in sections if section.strip()]

pdf_path = 'SOP_file.pdf'
pages_text = extract_text_from_pdf(pdf_path)
relevant_pages = extract_relevant_pages(pages_text)

classified_sections = []

for page_num, page_text in enumerate(relevant_pages, start=1):
    cleaned_text = remove_formatting(page_text)
    clean_text = remove_punctuation(cleaned_text)
    sections = split_into_sections(clean_text)

    for section in sections:
        if section.strip():
            prediction = classify_text(section, model, tokenizer, device)
            if prediction == 1:
                classified_sections.append(f"Page {page_num}:\n{section}")

output_file_path = 'High Potential Violation Directives.txt'
with open(output_file_path, 'w') as file:
    for section_text in classified_sections:
        file.write(section_text + "\n\n")

print(f"Classified sections saved to {output_file_path}")
