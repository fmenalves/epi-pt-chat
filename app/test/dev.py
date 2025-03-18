from app.core import present_result_filtered

medication = "Ciplox MG"
strength = "500 mg"
question = "Quais são as principais indicações terapêuticas do Ciplox?"
answer = present_result_filtered(question, medication, strength)
print(answer)
