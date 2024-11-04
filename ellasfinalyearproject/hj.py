import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate response in Igbo
def generate_response(user_input):
    if user_input.lower() == "kee ka ị mee?" or user_input.lower() == "nnọọ":
         return "di mụ mma"
    elif user_input.lower() == "ndeewo":
         return "Nnọọ nụ!"
    elif user_input.lower() == "kedụ?":
        return "O di mma"
    elif user_input.lower() == "o me mma" or user_input.lower() == "chineke":
        return "O gini"
    elif user_input.lower() =="chelu nu":
        return "O ginidi"
    elif user_input.lower() == "aha m bu" or user_input.lower() == "nsogbu adịghịị":
        return "O joka"
    elif user_input.lower() == "kedu aha gi":
        return "afa m bu"
    elif user_input.lower() == "ị na-asụ igbo?":
        return "Eee,a na m asu obele igbo"
    elif user_input.lower() == "hapụ m aka!" or user_input.lower() == "nyere m aka!" or user_input.lower() == "nodi ani" or user_input.lower() == "lee anya" or user_input.lower() == "dị ka enye" or user_input.lower() == "bụ" or user_input.lower() == "n’ihi na nke" or user_input.lower() == "anyị" or user_input.lower() == "ike" or user_input.lower() == "si" or user_input.lower() == "ọzọ ndị" or user_input.lower() == "nke" or user_input.lower() == "eme ha" or user_input.lower() == "oge" or user_input.lower() == "ma ọ bụrụ na" or user_input.lower() == "ekpe otú" or user_input.lower() == "kwuru" or user_input.lower() == "ihe" or user_input.lower() == "ọ bụla":
        return "Nyere m aka!, Ndo, Hapụ m aka!, Ka ọ dị echi, ndụ, ebe mgbe, azụ naanị, gburugburu n’akụkụ, mbụ nwoke, n’ebe ahụ, mgbe elu, iji, ụzọ banyere, ọtụtụ"

while True:
    user_input = input("Biko gosi ma ihe a nke o si ịchọ: ").strip('\"')  # Please input what you want to say:
    print("User input:", user_input)  # Print user input for debugging
    if user_input.lower() == 'kpọọ':
        print("Ka ọ dị. Ka omesịa!")  # Goodbye. See you later!
    
    elif user_input.lower() == "kedu":
        print("Odin ma!")  # Hi!
    
    else:
        response = generate_response(user_input)
        if response:
            print("Ihe mgbaghara a niile:", response)
        else:
            # Assuming you have tokenizer and model defined elsewhere
            input_ids = tokenizer.encode(user_input, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(input_ids, max_length=30, pad_token_id=tokenizer.eos_token_id)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print("Ihe mgbaghara a niile:", generated_text)  # The response is:
