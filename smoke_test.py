from language_id.data.loaders import load_commonvoice_lid
from language_id.models.classical.langdetect_model import LangdetectModel

# df = load_commonvoice_lid("dev").sample(200, random_state=0)
# m = LangdetectModel()
# preds = m.predict_batch(df["sentence"].tolist())
# print(sum(p.lang_code == g for p, g in zip(preds, df["lang"])) / len(df))

import dotenv
from any_llm import completion

# Load environment variables
dotenv.load_dotenv()

# Make a test request
r = completion(
    model='google/gemma-4-31B-it',
    provider='together',
    messages=[{'role': 'user', 'content': 'Identify the language: Bonjour le monde'}],
)
print(r.choices[0].message.content)

