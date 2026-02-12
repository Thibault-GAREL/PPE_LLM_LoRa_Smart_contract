"""
convert_to_ollama.py
Script pour convertir le modèle fine-tuné en format GGUF pour Ollama
"""

import os
import shutil
import subprocess
import json

# Configuration
MODEL_LORA_PATH = "./llama_lora_solidity_finetuned/final_model"
MODEL_BASE = "meta-llama/Llama-3.2-3B"
OUTPUT_DIR = "./ollama_model"
MODELFILE_PATH = os.path.join(OUTPUT_DIR, "Modelfile")
GGUF_PATH = os.path.join(OUTPUT_DIR, "model.gguf")

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


def merge_lora_weights():
    """
    Fusionne les poids LoRA avec le modèle de base.
    """
    print("🔀 Fusion des poids LoRA avec le modèle de base...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    
    # Charger le modèle de base
    print(f"   Chargement du modèle de base : {MODEL_BASE}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Charger et fusionner LoRA
    print(f"   Chargement des poids LoRA : {MODEL_LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, MODEL_LORA_PATH)
    
    print("   Fusion en cours...")
    model = model.merge_and_unload()
    
    # Sauvegarder le modèle fusionné
    merged_path = os.path.join(OUTPUT_DIR, "merged_model")
    os.makedirs(merged_path, exist_ok=True)
    
    print(f"   Sauvegarde du modèle fusionné : {merged_path}")
    model.save_pretrained(merged_path)
    
    # Sauvegarder le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LORA_PATH)
    tokenizer.save_pretrained(merged_path)
    
    print("   ✅ Fusion terminée")
    return merged_path


def convert_to_gguf(merged_model_path):
    """
    Convertit le modèle fusionné en format GGUF.
    
    Note: Nécessite llama.cpp
    """
    print("\n📦 Conversion en format GGUF...")
    
    # Vérifier si llama.cpp est disponible
    llama_cpp_path = input(
        "Entrez le chemin vers llama.cpp (ou appuyez sur Entrée pour ignorer) : "
    ).strip()
    
    if not llama_cpp_path:
        print("   ⚠️  Conversion GGUF ignorée")
        print("   Pour convertir manuellement :")
        print(f"   1. Clonez llama.cpp : git clone https://github.com/ggerganov/llama.cpp")
        print(f"   2. Compilez : cd llama.cpp && make")
        print(f"   3. Convertissez : python convert.py {merged_model_path}")
        print(f"   4. Quantifiez : ./quantize {merged_model_path}/ggml-model-f16.gguf model.gguf q4_0")
        return None
    
    convert_script = os.path.join(llama_cpp_path, "convert.py")
    
    if not os.path.exists(convert_script):
        print(f"   ❌ Script de conversion non trouvé : {convert_script}")
        return None
    
    # Convertir en GGUF
    print("   Conversion en cours...")
    try:
        subprocess.run([
            "python",
            convert_script,
            merged_model_path,
            "--outtype", "f16",
            "--outfile", GGUF_PATH
        ], check=True)
        
        print(f"   ✅ GGUF créé : {GGUF_PATH}")
        return GGUF_PATH
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erreur lors de la conversion : {e}")
        return None


def create_modelfile():
    """
    Crée le Modelfile pour Ollama.
    """
    print("\n📝 Création du Modelfile pour Ollama...")
    
    # Template du prompt système
    labels_description = "\n".join([
        f"- {i} si le contrat a une vulnérabilité de type \"{desc}\""
        if i < 8 else f"- {i} si le contrat est normal (sans vulnérabilité)"
        for i, desc in LABELS_8.items()
    ])
    
    system_prompt = f"""Tu es un expert en sécurité des smart contracts Solidity. Ta tâche est d'analyser du code Solidity et d'identifier les vulnérabilités.

Pour chaque contrat, réponds UNIQUEMENT avec UN SEUL chiffre entre 0 et 8 :
{labels_description}

IMPORTANT : Réponds UNIQUEMENT avec le chiffre correspondant à la vulnérabilité détectée, rien d'autre."""
    
    # Créer le Modelfile
    modelfile_content = f"""FROM {GGUF_PATH if os.path.exists(GGUF_PATH) else './model.gguf'}

# Paramètres du modèle
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop \"<|endoftext|>\"
PARAMETER stop \"</s>\"

# Template du système
TEMPLATE \"\"\"{{{{ if .System }}}}{{{{ .System }}}}{{{{ end }}}}

Contrat Solidity à analyser :
{{{{ .Prompt }}}}

Réponse (un seul chiffre) :\"\"\"

# Prompt système
SYSTEM \"\"\"
{system_prompt}
\"\"\"
"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(MODELFILE_PATH, 'w') as f:
        f.write(modelfile_content)
    
    print(f"   ✅ Modelfile créé : {MODELFILE_PATH}")
    return MODELFILE_PATH


def create_ollama_model():
    """
    Crée le modèle dans Ollama.
    """
    print("\n🚀 Création du modèle dans Ollama...")
    
    model_name = input("Entrez le nom du modèle Ollama (ex: llama-solidity) : ").strip()
    
    if not model_name:
        model_name = "llama-solidity"
    
    try:
        # Créer le modèle
        subprocess.run([
            "ollama",
            "create",
            model_name,
            "-f",
            MODELFILE_PATH
        ], check=True)
        
        print(f"\n   ✅ Modèle créé : {model_name}")
        print(f"\n   Vous pouvez maintenant l'utiliser avec :")
        print(f"   ollama run {model_name}")
        
        return model_name
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erreur lors de la création : {e}")
        print("\n   Créez le modèle manuellement avec :")
        print(f"   ollama create {model_name} -f {MODELFILE_PATH}")
        return None
    except FileNotFoundError:
        print("   ❌ Ollama n'est pas installé ou n'est pas dans le PATH")
        print("\n   Installez Ollama depuis : https://ollama.ai")
        print(f"\n   Puis créez le modèle avec :")
        print(f"   ollama create {model_name} -f {MODELFILE_PATH}")
        return None


def test_ollama_model(model_name):
    """
    Teste le modèle Ollama créé.
    """
    if not model_name:
        return
    
    print(f"\n🧪 Test du modèle {model_name}...")
    
    test_contract = """pragma solidity ^0.4.0;
contract Vulnerable {
    mapping(address => uint) balances;
    
    function withdraw() public {
        uint amount = balances[msg.sender];
        msg.sender.call.value(amount)();
        balances[msg.sender] = 0;
    }
}"""
    
    print("\n   Code de test (Reentrancy) :")
    print("   " + test_contract.replace("\n", "\n   "))
    
    try:
        result = subprocess.run([
            "ollama",
            "run",
            model_name,
            test_contract
        ], capture_output=True, text=True, timeout=30)
        
        print(f"\n   Réponse du modèle : {result.stdout.strip()}")
        print("   Attendu : 5 (Reentrancy)")
        
    except subprocess.TimeoutExpired:
        print("   ⚠️  Timeout - le modèle met trop de temps à répondre")
    except Exception as e:
        print(f"   ❌ Erreur lors du test : {e}")


def create_usage_script():
    """
    Crée un script Python pour utiliser le modèle Ollama.
    """
    print("\n📄 Création du script d'utilisation...")
    
    script_content = '''"""
use_ollama_model.py
Script pour utiliser le modèle Ollama fine-tuné
"""

import requests
import json

MODEL_NAME = "llama-solidity"  # À modifier si vous avez choisi un autre nom
OLLAMA_URL = "http://localhost:11434/api/generate"

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


def classify_contract(code):
    """
    Classifie un contrat Solidity avec le modèle Ollama.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            'model': MODEL_NAME,
            'prompt': code,
            'stream': False
        },
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json().get('response', '').strip()
        
        # Extraire le chiffre
        for char in result:
            if char.isdigit() and int(char) in range(9):
                return int(char)
    
    return -1


if __name__ == "__main__":
    # Exemple d'utilisation
    test_code = """pragma solidity ^0.4.0;
contract Test {
    function withdraw() public {
        msg.sender.call.value(balance)();
        balance = 0;
    }
}"""
    
    print("Classification d'un contrat de test...")
    prediction = classify_contract(test_code)
    
    if prediction != -1:
        print(f"Résultat : {prediction} - {LABELS_8[prediction]}")
    else:
        print("Erreur de classification")
'''
    
    script_path = os.path.join(OUTPUT_DIR, "use_ollama_model.py")
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"   ✅ Script créé : {script_path}")


def main():
    """
    Fonction principale.
    """
    print("="*60)
    print("CONVERSION DU MODÈLE POUR OLLAMA")
    print("="*60)
    
    # Vérifier que le modèle LoRA existe
    if not os.path.exists(MODEL_LORA_PATH):
        print(f"\n❌ Modèle LoRA non trouvé : {MODEL_LORA_PATH}")
        print("   Entraînez d'abord le modèle avec fine_tune_llama_lora.py")
        return
    
    print(f"\n✅ Modèle LoRA trouvé : {MODEL_LORA_PATH}")
    
    # Étapes de conversion
    print("\n" + "="*60)
    print("ÉTAPES DE CONVERSION")
    print("="*60)
    print("1. Fusion des poids LoRA avec le modèle de base")
    print("2. Conversion en format GGUF (optionnel)")
    print("3. Création du Modelfile pour Ollama")
    print("4. Création du modèle dans Ollama")
    print("5. Test du modèle")
    
    response = input("\nContinuer ? (y/n): ")
    if response.lower() != 'y':
        print("Conversion annulée")
        return
    
    # 1. Fusionner les poids
    merged_path = merge_lora_weights()
    
    # 2. Convertir en GGUF (optionnel)
    convert_to_gguf(merged_path)
    
    # 3. Créer le Modelfile
    create_modelfile()
    
    # 4. Créer le modèle dans Ollama
    model_name = create_ollama_model()
    
    # 5. Tester le modèle
    if model_name:
        test_ollama_model(model_name)
    
    # Créer le script d'utilisation
    create_usage_script()
    
    print("\n" + "="*60)
    print("CONVERSION TERMINÉE")
    print("="*60)
    
    if model_name:
        print(f"\n✅ Modèle Ollama créé : {model_name}")
        print(f"\nPour l'utiliser :")
        print(f"  ollama run {model_name}")
        print(f"\nOu avec le script Python :")
        print(f"  python {OUTPUT_DIR}/use_ollama_model.py")
    else:
        print(f"\n⚠️  Modèle non créé dans Ollama")
        print(f"   Fichiers disponibles dans : {OUTPUT_DIR}")
        print(f"   Consultez le README pour la procédure manuelle")


if __name__ == "__main__":
    main()
