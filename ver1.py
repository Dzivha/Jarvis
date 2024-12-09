import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import dateparser
import datetime

# Load a pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the conversational pipeline
chat = pipeline("conversational", model=model, tokenizer=tokenizer)

# Calendar database (in-memory)
calendar = []

def add_event(event_description, date_time):
    event = {"description": event_description, "date": date_time}
    calendar.append(event)
    return f"Added event: {event_description} on {date_time.strftime('%Y-%m-%d %H:%M')}"

def list_events():
    if not calendar:
        return "No upcoming events."
    response = "Upcoming events:\n"
    for event in calendar:
        response += f"- {event['description']} at {event['date'].strftime('%Y-%m-%d %H:%M')}\n"
    return response

def handle_command(command):
    # Use the model to predict the intent
    response = chat(tokenizer.encode(command, return_tensors="pt"))
    response_text = tokenizer.decode(response.generated_responses[0], skip_special_tokens=True)
    print("AI thinks you said: ", response_text)
    
    # Parse dates and commands
    if "add event" in response_text.lower():
        date_time = dateparser.parse(' '.join(response_text.split()[3:]))
        if date_time:
            return add_event(' '.join(response_text.split()[2:3]), date_time)
        else:
            return "I couldn't understand the date and time."
    elif "show" in response_text.lower() and "events" in response_text.lower():
        return list_events()
    else:
        return "Sorry, I didn't understand that."

# Example usage
print(handle_command("Can you add an event meeting with John on next Monday at 3 PM?"))
print(handle_command("Show me the events"))
