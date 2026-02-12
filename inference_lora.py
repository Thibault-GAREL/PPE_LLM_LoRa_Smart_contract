"""
inference_lora.py
Script pour utiliser le modèle LLaMA fine-tuné avec LoRA
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd

# Configuration
MODEL_BASE = "meta-llama/Llama-3.2-3B"  # Doit correspondre au modèle de base
MODEL_LORA = "./llama_lora_solidity_finetuned/final_model"  # Chemin vers le modèle LoRA

LABELS_8 = {
    0: "Block number dependency (BN)",
    1: "Dangerous delegatecall (DE)",
    2: "Ether frozen (EF)",
    3: "Ether strict equality (SE)",
    4: "Integer overflow (OF)",
    5: "Reentrancy (RE)",
    6: "Timestamp dependency (TP)",
    7: "Unchecked external call (UC)",
    8: "Normal"
}


def load_finetuned_model(base_model_name, lora_model_path):
    """
    Charge le modèle de base et applique les poids LoRA.
    """
    print(f"🤖 Chargement du modèle de base : {base_model_name}")
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
    
    # Charger le modèle de base
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Charger les poids LoRA
    print(f"⚙️  Application des poids LoRA depuis : {lora_model_path}")
    model = PeftModel.from_pretrained(model, lora_model_path)
    
    # Merger les poids pour accélérer l'inférence (optionnel)
    # model = model.merge_and_unload()
    
    model.eval()
    print(f"✅ Modèle chargé avec succès\n")
    
    return model, tokenizer


def create_prompt(code):
    """
    Crée le prompt pour la classification.
    """
    labels_description = "\n".join([
        f"- {i} si le contrat a une vulnérabilité de type \"{desc}\""
        if i < 8 else f"- {i} si le contrat est normal (sans vulnérabilité)"
        for i, desc in LABELS_8.items()
    ])
    
    prompt = f"""Analyse ce contrat Solidity et identifie s'il contient une vulnérabilité.

Réponds UNIQUEMENT avec UN SEUL chiffre entre 0 et 8 :
{labels_description}

IMPORTANT : Réponds UNIQUEMENT avec le chiffre, rien d'autre.

Contrat Solidity à analyser :
{code}

Réponse (un seul chiffre) :"""
    
    return prompt


def classify_contract(model, tokenizer, code, max_length=2048):
    """
    Classifie un contrat Solidity.
    """
    prompt = create_prompt(code)
    
    # Tokenizer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Générer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Décoder
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    
    # Extraire le chiffre prédit
    for char in response:
        if char.isdigit() and int(char) in range(9):
            return int(char)
    
    return -1  # Erreur


def classify_dataset(model, tokenizer, csv_path, output_path=None):
    """
    Classifie tous les contrats d'un CSV.
    """
    print(f"📁 Chargement du dataset : {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"🔄 Classification de {len(df)} contrats...\n")
    
    predictions = []
    
    for idx, row in df.iterrows():
        prediction = classify_contract(model, tokenizer, row['code'])
        predictions.append(prediction)
        
        if (idx + 1) % 10 == 0:
            print(f"Progression : {idx + 1}/{len(df)}")
    
    df['llm_prediction'] = predictions
    
    # Calculer la précision si les labels sont disponibles
    if 'label_encoded' in df.columns:
        df['label_encoded'] = pd.to_numeric(df['label_encoded'], errors='coerce')
        df_valid = df[df['llm_prediction'] != -1]
        correct = (df_valid['label_encoded'] == df_valid['llm_prediction']).sum()
        accuracy = correct / len(df_valid) * 100
        
        print(f"\n✅ Précision : {accuracy:.2f}%")
        print(f"Prédictions correctes : {correct}/{len(df_valid)}")
    
    # Sauvegarder les résultats
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n💾 Résultats sauvegardés dans : {output_path}")
    
    return df


def interactive_mode(model, tokenizer):
    """
    Mode interactif pour tester le modèle.
    """
    print("\n" + "="*60)
    print("MODE INTERACTIF")
    print("="*60)
    print("Collez votre code Solidity (tapez 'END' sur une nouvelle ligne pour terminer)")
    print("Tapez 'quit' pour quitter\n")
    
    while True:
        print("\n" + "-"*60)
        print("Entrez le code Solidity :")
        
        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            if line.strip().lower() == 'quit':
                print("\nAu revoir !")
                return
            lines.append(line)
        
        code = "\n".join(lines)
        
        if not code.strip():
            print("⚠️  Code vide, réessayez")
            continue
        
        print("\n🔍 Analyse en cours...")
        prediction = classify_contract(model, tokenizer, code)
        
        if prediction == -1:
            print("❌ Erreur de classification")
        else:
            print(f"\n✅ Résultat : {prediction} - {LABELS_8[prediction]}")


# ========================================
# PROGRAMME PRINCIPAL
# ========================================
if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("INFÉRENCE AVEC LE MODÈLE FINE-TUNÉ")
    print("="*60)
    
    # Charger le modèle
    model, tokenizer = load_finetuned_model(MODEL_BASE, MODEL_LORA)
    
    # Vérifier les arguments
    if len(sys.argv) > 1:
        # Mode batch : classifier un CSV
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "predictions_finetuned.csv"
        classify_dataset(model, tokenizer, csv_path, output_path)
    else:
        # Mode interactif
        interactive_mode(model, tokenizer)
