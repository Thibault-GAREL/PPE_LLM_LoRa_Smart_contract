"""
colab_finetune.py
Script de fine-tuning pour Google Colab
À exécuter APRÈS avoir installé les dépendances et uploadé le CSV
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
print("FINE-TUNING LLAMA AVEC LoRA - GOOGLE COLAB")
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

# Chercher dans différents emplacements possibles
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
    # Essayer de charger avec des paramètres tolérants
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

# Vérifier les colonnes requises
if 'code' not in df.columns or 'label_encoded' not in df.columns:
    print(f"❌ Colonnes manquantes ! Colonnes présentes : {list(df.columns)}")
    sys.exit(1)

# Supprimer les valeurs manquantes
df = df.dropna(subset=['code', 'label_encoded'])
df = df[df['code'].str.strip() != '']

# Convertir les labels
df['label_encoded'] = pd.to_numeric(df['label_encoded'], errors='coerce')
df = df.dropna(subset=['label_encoded'])
df['label_encoded'] = df['label_encoded'].astype(int)

# Garder seulement les labels valides (0-8)
df = df[df['label_encoded'].isin(range(9))]

print(f"   Lignes initiales : {initial_count}")
print(f"   Lignes après nettoyage : {len(df)}")
print(f"   Lignes supprimées : {initial_count - len(df)}")

# Sauvegarder le CSV nettoyé
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "dataset_path": cleaned_path,
    "test_size": 0.2,
    "random_state": 42,
    "max_samples": 1000,  # Limité pour Colab (session 12h)
    
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    
    "output_dir": "./llama_lora_colab",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    
    "logging_steps": 25,
    "save_steps": 100,
    "eval_steps": 100,
    
    "max_length": 1024,
    "use_4bit": True,
}

print("\n" + "="*70)
print("CONFIGURATION")
print("="*70)
print(f"Modèle : {CONFIG['model_name']}")
print(f"Dataset : {len(df)} contrats (limité à {CONFIG['max_samples']})")
print(f"Époques : {CONFIG['num_train_epochs']}")
print(f"Batch size : {CONFIG['per_device_train_batch_size']}")
print(f"Max length : {CONFIG['max_length']}")
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
    
    # Limiter si nécessaire
    if max_samples and len(df) > max_samples:
        print(f"   ⚠️  Limitation à {max_samples} contrats pour économiser du temps")
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    
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
    print("   Cela peut prendre 2-3 minutes...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
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
    
    # Arguments d'entraînement
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
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )
    
    print("\n   🚀 Lancement de l'entraînement...")
    print(f"   ⏱️  Durée estimée : 2-3 heures")
    print("   💡 Vous pouvez fermer cet onglet, Colab continuera")
    print()
    
    trainer.train()
    
    print("\n   ✅ Entraînement terminé !")
    
    # Sauvegarder
    final_path = os.path.join(config["output_dir"], "final_model")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"   💾 Modèle sauvegardé : {final_path}")
    
    return trainer, final_path


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

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ !")
print("="*70)
print(f"\n📁 Modèle sauvegardé dans : {final_path}")
print("\n💡 Pour télécharger le modèle, exécutez dans une cellule Colab :")
print("   from google.colab import files")
print("   import shutil")
print(f"   shutil.make_archive('model', 'zip', '{CONFIG['output_dir']}')")
print("   files.download('model.zip')")
