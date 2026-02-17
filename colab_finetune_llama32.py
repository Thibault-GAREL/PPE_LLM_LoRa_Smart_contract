"""
colab_finetune_llama32.py
Script de fine-tuning LLaMA 3.2 3B pour Google Colab
Version complète avec tous les contrats (pas de limitation)
"""

import os
import sys
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from sklearn.model_selection import train_test_split
import json

print("="*70)
print("FINE-TUNING LLAMA 3.2 3B AVEC LoRA - GOOGLE COLAB")
print("="*70)

# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : Vérifier le GPU
# ═══════════════════════════════════════════════════════════════════════

print("\n🔍 Vérification du GPU...")
if not torch.cuda.is_available():
    print("❌ ERREUR : GPU non détecté !")
    print("\n⚠️  ACTIVEZ LE GPU dans Colab :")
    print("   1. Menu : Runtime → Change runtime type")
    print("   2. Hardware accelerator : GPU (T4)")
    print("   3. Save")
    print("   4. Relancez ce script")
    sys.exit(1)

print(f"✅ GPU disponible : {torch.cuda.get_device_name(0)}")
print(f"   Mémoire VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : Trouver le fichier CSV
# ═══════════════════════════════════════════════════════════════════════

print("\n📁 Recherche du fichier CSV...")

possible_paths = [
    'SC_Vuln_8label.csv',
    'archive/SC_Vuln_8label.csv',
    '/content/SC_Vuln_8label.csv',
    '/content/archive/SC_Vuln_8label.csv'
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        print(f"✅ CSV trouvé : {csv_path}")
        break

if csv_path is None:
    print("❌ Fichier CSV non trouvé !")
    print("\n⚠️  UPLOADEZ le fichier CSV d'abord :")
    print("   Dans une cellule Colab, exécutez :")
    print("   from google.colab import files")
    print("   files.upload()")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : Nettoyer et charger le CSV
# ═══════════════════════════════════════════════════════════════════════

print("\n🔧 Chargement et nettoyage du CSV...")

try:
    df = pd.read_csv(
        csv_path,
        on_bad_lines='skip',
        engine='python',
        encoding='utf-8'
    )
    print(f"✅ CSV chargé : {len(df)} lignes")
except Exception as e:
    print(f"⚠️  Première tentative échouée, essai avec latin-1...")
    df = pd.read_csv(
        csv_path,
        on_bad_lines='skip',
        engine='python',
        encoding='latin-1'
    )
    print(f"✅ CSV chargé : {len(df)} lignes")

# Nettoyer les données
print("\n🧹 Nettoyage des données...")
initial_count = len(df)

if 'code' not in df.columns or 'label_encoded' not in df.columns:
    print(f"❌ Colonnes manquantes ! Colonnes présentes : {list(df.columns)}")
    sys.exit(1)

df = df.dropna(subset=['code', 'label_encoded'])
df = df[df['code'].str.strip() != '']
df['label_encoded'] = pd.to_numeric(df['label_encoded'], errors='coerce')
df = df.dropna(subset=['label_encoded'])
df['label_encoded'] = df['label_encoded'].astype(int)
df = df[df['label_encoded'].isin(range(9))]

print(f"   Lignes initiales : {initial_count}")
print(f"   Lignes après nettoyage : {len(df)}")
print(f"   Lignes supprimées : {initial_count - len(df)}")

cleaned_path = 'SC_Vuln_8label_cleaned.csv'
df.to_csv(cleaned_path, index=False)
print(f"✅ CSV nettoyé sauvegardé : {cleaned_path}")

# Afficher la distribution
print("\n📊 Distribution des labels :")
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

for label, count in df['label_encoded'].value_counts().sort_index().items():
    print(f"   {label} ({LABELS_8[label]}): {count}")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION - VERSION COMPLÈTE POUR COLAB
# ═══════════════════════════════════════════════════════════════════════

CONFIG = {
    # Modèle complet LLaMA 3.2 3B
    "model_name": "meta-llama/Llama-3.2-3B",
    
    # Dataset complet (pas de limitation)
    "dataset_path": cleaned_path,
    "test_size": 0.2,
    "random_state": 42,
    "max_samples": None,  # None = utiliser TOUS les contrats
    
    # LoRA parameters (version complète)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    
    # Training parameters
    "output_dir": "./llama32_lora_full",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    
    "max_length": 2048,  # Longueur maximale complète
    "use_4bit": True,
}

print("\n" + "="*70)
print("CONFIGURATION")
print("="*70)
print(f"Modèle : {CONFIG['model_name']}")
print(f"Dataset : {len(df)} contrats (TOUS utilisés)")
print(f"Époques : {CONFIG['num_train_epochs']}")
print(f"Batch size : {CONFIG['per_device_train_batch_size']}")
print(f"Max length : {CONFIG['max_length']}")
print(f"LoRA rank : {CONFIG['lora_r']}")
print("="*70)

# ═══════════════════════════════════════════════════════════════════════
# FONCTIONS
# ═══════════════════════════════════════════════════════════════════════

def create_prompt(code, label=None):
    """Crée le prompt pour l'entraînement."""
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
    
    if label is not None:
        prompt += f" {label}"
    
    return prompt


def load_and_prepare_data(df_input, test_size=0.2, random_state=42, max_samples=None):
    """Prépare le dataset."""
    print(f"\n📊 Préparation du dataset...")
    
    df = df_input.copy()
    
    # Limiter si nécessaire (mais par défaut on utilise tout)
    if max_samples and len(df) > max_samples:
        print(f"   ⚠️  Limitation à {max_samples} contrats")
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    else:
        print(f"   ✅ Utilisation de TOUS les {len(df)} contrats")
    
    # Créer les prompts
    print("   Création des prompts...")
    df['text'] = df.apply(
        lambda row: create_prompt(row['code'], row['label_encoded']),
        axis=1
    )
    
    # Split train/test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label_encoded']
    )
    
    print(f"   Train : {len(train_df)} contrats")
    print(f"   Test  : {len(test_df)} contrats")
    
    train_dataset = Dataset.from_pandas(train_df[['text']])
    test_dataset = Dataset.from_pandas(test_df[['text']])
    
    return train_dataset, test_dataset, test_df


def load_model_and_tokenizer(model_name, use_4bit=True):
    """Charge le modèle et tokenizer."""
    print(f"\n🤖 Chargement du modèle : {model_name}")
    print("   Cela peut prendre 3-5 minutes...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"\n⚠️  Erreur lors du chargement de {model_name}")
        print(f"   Erreur : {e}")
        print("\n💡 Essai avec un modèle alternatif : TinyLlama...")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print("   ✅ Modèle chargé")
    return model, tokenizer


def setup_lora(model, config):
    """Configure LoRA."""
    print("\n⚙️  Configuration de LoRA...")
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"   Paramètres entraînables : {trainable:,}")
    print(f"   Tous les paramètres : {total:,}")
    print(f"   Pourcentage entraînable : {100*trainable/total:.2f}%")
    
    return model


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize les exemples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )


def train_model(model, tokenizer, train_dataset, test_dataset, config):
    """Entraîne le modèle."""
    print("\n🏋️  Début de l'entraînement...")
    
    # Tokenizer
    print("   Tokenization des données...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Arguments d'entraînement (CORRECTION : eval_strategy au lieu de evaluation_strategy)
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        eval_strategy="steps",  # ✅ CORRECTION ICI
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
        save_total_limit=3,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )
    
    print("\n   🚀 Lancement de l'entraînement...")
    print(f"   ⏱️  Durée estimée : 3-6 heures (dataset complet)")
    print(f"   💡 Nombre d'étapes : ~{len(tokenized_train) * config['num_train_epochs'] // (config['per_device_train_batch_size'] * config['gradient_accumulation_steps'])}")
    print("   📊 Vous pouvez fermer cet onglet, Colab continuera")
    print()
    
    trainer.train()
    
    print("\n   ✅ Entraînement terminé !")
    
    # Sauvegarder
    final_path = os.path.join(config["output_dir"], "final_model")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"   💾 Modèle sauvegardé : {final_path}")
    
    return trainer, final_path


def evaluate_quick(model, tokenizer, test_df, max_samples=50):
    """Évaluation rapide sur quelques échantillons."""
    print(f"\n📊 Évaluation rapide sur {max_samples} échantillons...")
    
    model.eval()
    predictions = []
    true_labels = []
    
    for idx, row in test_df.head(max_samples).iterrows():
        prompt = create_prompt(row['code'])
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        predicted_label = -1
        for char in response:
            if char.isdigit() and int(char) in range(9):
                predicted_label = int(char)
                break
        
        predictions.append(predicted_label)
        true_labels.append(row['label_encoded'])
    
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(predictions) * 100
    
    print(f"   ✅ Précision : {accuracy:.2f}% ({correct}/{len(predictions)})")
    
    return accuracy


# ═══════════════════════════════════════════════════════════════════════
# EXÉCUTION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("DÉMARRAGE DU FINE-TUNING")
print("="*70)

# 1. Préparer les données
train_dataset, test_dataset, test_df = load_and_prepare_data(
    df,
    CONFIG["test_size"],
    CONFIG["random_state"],
    CONFIG["max_samples"]
)

# 2. Charger le modèle
model, tokenizer = load_model_and_tokenizer(
    CONFIG["model_name"],
    CONFIG["use_4bit"]
)

# 3. Setup LoRA
model = setup_lora(model, CONFIG)

# 4. Sauvegarder la config
os.makedirs(CONFIG["output_dir"], exist_ok=True)
with open(os.path.join(CONFIG["output_dir"], "config.json"), 'w') as f:
    json.dump(CONFIG, f, indent=2)

# 5. Entraîner
trainer, final_path = train_model(model, tokenizer, train_dataset, test_dataset, CONFIG)

# 6. Évaluation rapide
accuracy = evaluate_quick(model, tokenizer, test_df, max_samples=50)

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ !")
print("="*70)
print(f"\n📁 Modèle sauvegardé : {final_path}")
print(f"📊 Précision (échantillon) : {accuracy:.2f}%")
print("\n💡 Pour télécharger le modèle, exécutez dans une cellule Colab :")
print("   from google.colab import files")
print("   import shutil")
print(f"   shutil.make_archive('llama32_finetuned', 'zip', '{CONFIG['output_dir']}')")
print("   files.download('llama32_finetuned.zip')")
