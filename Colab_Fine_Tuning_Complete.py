# ═══════════════════════════════════════════════════════════════════════
# NOTEBOOK GOOGLE COLAB - FINE-TUNING LLAMA AVEC LoRA
# Classification de vulnérabilités Solidity
# ═══════════════════════════════════════════════════════════════════════

"""
INSTRUCTIONS AVANT DE COMMENCER :

1. ACTIVER LE GPU (TRÈS IMPORTANT) :
   - Menu : Runtime → Change runtime type
   - Hardware accelerator : GPU (T4)
   - Cliquez sur Save
   - ⚠️ Sans cela, l'entraînement sera IMPOSSIBLE

2. VÉRIFIER que le GPU est activé :
   - Exécutez la cellule ci-dessous
   - Vous devriez voir : "GPU disponible : Tesla T4"
"""

# ═══════════════════════════════════════════════════════════════════════
# CELLULE 1 : Vérifier le GPU
# ═══════════════════════════════════════════════════════════════════════

import torch
print("🔍 Vérification du GPU...")
if torch.cuda.is_available():
    print(f"✅ GPU disponible : {torch.cuda.get_device_name(0)}")
    print(f"   Mémoire VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ ERREUR : GPU non détecté !")
    print("\n⚠️  SOLUTION : Activez le GPU :")
    print("   1. Menu : Runtime → Change runtime type")
    print("   2. Hardware accelerator : GPU (T4)")
    print("   3. Save")
    print("   4. Réexécutez cette cellule")
    raise SystemExit("GPU non activé")

print("\n✅ Tout est OK, vous pouvez continuer !")


# ═══════════════════════════════════════════════════════════════════════
# CELLULE 2 : Installation des dépendances
# ═══════════════════════════════════════════════════════════════════════

print("\n📦 Installation des dépendances...")
print("⏱️  Cela prend environ 2-3 minutes...\n")

!pip install -q transformers>=4.35.0
!pip install -q peft>=0.7.0
!pip install -q bitsandbytes>=0.41.0
!pip install -q accelerate>=0.24.0
!pip install -q datasets>=2.14.0
!pip install -q scikit-learn

print("\n✅ Installation terminée !")


# ═══════════════════════════════════════════════════════════════════════
# CELLULE 3 : Upload du fichier CSV
# ═══════════════════════════════════════════════════════════════════════

from google.colab import files
import os
import pandas as pd

print("📁 Upload de votre fichier CSV...")
print("⚠️  Cliquez sur 'Choisir un fichier' et sélectionnez SC_Vuln_8label.csv\n")

uploaded = files.upload()

# Trouver le fichier CSV uploadé
csv_file = None
for filename in uploaded.keys():
    if filename.endswith('.csv'):
        csv_file = filename
        break

if csv_file is None:
    raise ValueError("❌ Aucun fichier CSV trouvé. Veuillez uploader SC_Vuln_8label.csv")

print(f"\n✅ Fichier uploadé : {csv_file}")

# Créer le dossier archive si nécessaire
os.makedirs('archive', exist_ok=True)

# Déplacer le fichier
os.rename(csv_file, f'archive/{csv_file}')
print(f"✅ Fichier déplacé vers archive/{csv_file}")


# ═══════════════════════════════════════════════════════════════════════
# CELLULE 4 : Nettoyer le CSV (corriger les erreurs de parsing)
# ═══════════════════════════════════════════════════════════════════════

print("\n🔧 Nettoyage du fichier CSV...")

csv_path = f'archive/{csv_file}'

# Lire le CSV avec des paramètres plus permissifs
try:
    df = pd.read_csv(
        csv_path,
        on_bad_lines='skip',  # Ignorer les lignes problématiques
        engine='python',      # Utiliser le parser Python (plus tolérant)
        encoding='utf-8',
        quoting=1             # QUOTE_ALL
    )
    print(f"✅ CSV chargé : {len(df)} lignes")
except Exception as e:
    print(f"❌ Erreur : {e}")
    print("\n🔄 Tentative avec un autre encodage...")
    df = pd.read_csv(
        csv_path,
        on_bad_lines='skip',
        engine='python',
        encoding='latin-1',
        quoting=1
    )
    print(f"✅ CSV chargé : {len(df)} lignes")

# Nettoyer les données
print("\n🧹 Nettoyage des données...")
initial_count = len(df)

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna(subset=['code', 'label_encoded'])

# Supprimer les lignes avec du code vide
df = df[df['code'].str.strip() != '']

# Convertir les labels en entiers
df['label_encoded'] = pd.to_numeric(df['label_encoded'], errors='coerce')
df = df.dropna(subset=['label_encoded'])
df['label_encoded'] = df['label_encoded'].astype(int)

# Garder seulement les labels valides (0-8)
df = df[df['label_encoded'].isin(range(9))]

print(f"   Lignes initiales : {initial_count}")
print(f"   Lignes après nettoyage : {len(df)}")
print(f"   Lignes supprimées : {initial_count - len(df)}")

# Sauvegarder le CSV nettoyé
cleaned_path = 'archive/SC_Vuln_8label_cleaned.csv'
df.to_csv(cleaned_path, index=False)
print(f"\n✅ CSV nettoyé sauvegardé : {cleaned_path}")

# Afficher la distribution
print("\n📊 Distribution des labels :")
for label, count in df['label_encoded'].value_counts().sort_index().items():
    print(f"   {label}: {count} contrats")


# ═══════════════════════════════════════════════════════════════════════
# CELLULE 5 : Script de fine-tuning (VERSION COLAB OPTIMISÉE)
# ═══════════════════════════════════════════════════════════════════════

import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from sklearn.model_selection import train_test_split
import json

# Configuration optimisée pour Colab
CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Modèle léger
    "dataset_path": cleaned_path,  # Utiliser le CSV nettoyé
    "test_size": 0.2,
    "random_state": 42,
    "max_samples": 1000,  # Limité pour Colab (12h max)
    
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    
    # Training
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
    
    "max_length": 1024,  # Réduit pour Colab
    "use_4bit": True,
}

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

print("\n" + "="*60)
print("CONFIGURATION")
print("="*60)
for key, value in CONFIG.items():
    if key not in ['target_modules']:
        print(f"{key}: {value}")
print("="*60)


def create_prompt(code, label=None):
    """Crée le prompt."""
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


def load_and_prepare_data(csv_path, test_size=0.2, random_state=42, max_samples=None):
    """Charge et prépare le dataset."""
    print(f"\n📁 Chargement du dataset : {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Contrats : {len(df)}")
    
    # Limiter si nécessaire
    if max_samples and len(df) > max_samples:
        print(f"   ⚠️  Limitation à {max_samples} contrats")
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    
    # Créer les prompts
    print("   Création des prompts...")
    df['text'] = df.apply(
        lambda row: create_prompt(row['code'], row['label_encoded']),
        axis=1
    )
    
    # Split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label_encoded']
    )
    
    print(f"\n   Train : {len(train_df)} | Test : {len(test_df)}")
    
    train_dataset = Dataset.from_pandas(train_df[['text']])
    test_dataset = Dataset.from_pandas(test_df[['text']])
    
    return train_dataset, test_dataset, test_df


def load_model_and_tokenizer(model_name, use_4bit=True):
    """Charge le modèle."""
    print(f"\n🤖 Chargement : {model_name}")
    
    from transformers import BitsAndBytesConfig
    
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
    
    print("   ✅ Chargé")
    return model, tokenizer


def setup_lora(model, config):
    """Configure LoRA."""
    print("\n⚙️  Configuration LoRA...")
    
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
    
    print(f"   Entraînables : {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )


def train_model(model, tokenizer, train_dataset, test_dataset, config):
    """Entraîne."""
    print("\n🏋️  Entraînement...")
    
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
    
    print("   🚀 Début...")
    trainer.train()
    
    print("\n   ✅ Terminé !")
    
    # Sauvegarder
    final_path = os.path.join(config["output_dir"], "final_model")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"   💾 Sauvegardé : {final_path}")
    
    return trainer


# ═══════════════════════════════════════════════════════════════════════
# CELLULE 6 : Lancer l'entraînement
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("DÉMARRAGE DU FINE-TUNING")
print("="*60)

# 1. Charger les données
train_dataset, test_dataset, test_df = load_and_prepare_data(
    CONFIG["dataset_path"],
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

# 4. Entraîner
trainer = train_model(model, tokenizer, train_dataset, test_dataset, CONFIG)

print("\n" + "="*60)
print("✅ ENTRAÎNEMENT TERMINÉ !")
print("="*60)
print(f"\nModèle sauvegardé dans : {CONFIG['output_dir']}/final_model")


# ═══════════════════════════════════════════════════════════════════════
# CELLULE 7 : Télécharger le modèle fine-tuné
# ═══════════════════════════════════════════════════════════════════════

print("\n📦 Compression du modèle pour téléchargement...")

import shutil

# Créer une archive
output_zip = 'llama_finetuned_model'
shutil.make_archive(output_zip, 'zip', CONFIG['output_dir'])

print(f"✅ Archive créée : {output_zip}.zip")

# Télécharger
print("\n⬇️  Téléchargement de l'archive...")
files.download(f'{output_zip}.zip')

print("\n✅ Téléchargement terminé !")
print("\nVous pouvez maintenant utiliser ce modèle en local avec inference_lora.py")
