import google.generativeai as genai

def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-pro')
    except Exception as e:
        return None

def generate_initial_report(vision_model, img, predicted_label, confidence, patient_info, audience_mode, language):
    
    tone_instruction = "Use simple, empathetic language suitable for a patient without medical training." if audience_mode == "Patient" else "Use precise, clinical medical terminology suitable for a doctor's chart."
    
    prompt = f"""
    You are an expert orthopedic AI assistant. 
    A custom CNN model has evaluated the provided knee X-ray and predicted it as: '{predicted_label}' with {confidence:.2f}% confidence.
    
    Patient Profile:
    - Age: {patient_info['age']}
    - Gender: {patient_info['gender']}
    - Symptoms & History: {patient_info['history']}
    
    Tasks:
    1. Explain what a diagnosis of '{predicted_label}' means medically in the context of knee osteoarthritis.
    2. Relate this directly to the patient's age and submitted symptoms.
    3. Suggest appropriate clinical next steps, lifestyle modifications, or precautions.
    
    Instructions:
    - {tone_instruction}
    - Provide the final output strictly in the {language} language.
    - Keep the response professional and structured. Add a disclaimer that this is AI-assisted.
    """
    
    response = vision_model.generate_content([prompt, img])
    return response.text
    
def chat_with_image_context(vision_model, img, chat_history, new_message):
    # Construct conversation history as a prompt context
    context = "Here is the conversation history so far regarding the provided medical image:\n"
    for msg in chat_history:
        context += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    context += f"\nUser's new question: {new_message}\nAnswer professionally based on the image and context."
    
    response = vision_model.generate_content([context, img])
    return response.text