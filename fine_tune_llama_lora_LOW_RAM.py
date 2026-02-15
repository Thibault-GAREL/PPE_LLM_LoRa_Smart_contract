"""
fine_tune_llama_lora_LOW_RAM.py
Version OPTIMISÉE pour machines avec peu de RAM/VRAM
"""

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
import gc

# ========================================
# CONFIGURATION OPTIMISÉE POUR LOW RAM
# ========================================
CONFIG = {
    # ⚠️ MODÈLE PLUS LÉGER - Choisissez selon votre RAM :
    # Option 1 : TinyLlama (1GB VRAM minimum) - RECOMMANDÉ
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    # Option 2 : Phi-2 (4GB VRAM)
    # "model_name": "microsoft/phi-2",
    
    # Option 3 : LLaMA 3.2 1B (2-3GB VRAM)
    # "model_name": "meta-llama/Llama-3.2-1B",
    
    # Dataset - LIMITÉ pour tester
    "dataset_path": "archive/SC_Vuln_8label.csv",
    "test_size": 0.2,
    "random_state": 42,
    "max_samples": 500,  # ⚠️ LIMITÉ à 500 pour économiser RAM
    
    # LoRA parameters - RÉDUITS
    "lora_r": 8,               # ⬇️ Réduit de 16 à 8
    "lora_alpha": 16,          # ⬇️ Réduit de 32 à 16
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],  # ⚠️ Seulement 2 modules au lieu de 4
    
    # Training parameters - OPTIMISÉS
    "output_dir": "./llama_lora_solidity_lowram",
    "num_train_epochs": 2,     # ⬇️ Réduit de 3 à 2
    "per_device_train_batch_size": 1,  # ⚠️ BATCH SIZE = 1
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,  # ⬆️ Augmenté pour compenser
    "learning_rate": 2e-4,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    
    # Logging - ESPACÉ pour économiser
    "logging_steps": 50,       # ⬆️ Augmenté
    "save_steps": 200,         # ⬆️ Augmenté
    "eval_steps": 200,         # ⬆️ Augmenté
    
    # Autres - OPTIMISÉS
    "max_length": 1024,        # ⬇️ RÉDUIT de 2048 à 1024
    "use_4bit": True,
    "use_8bit": False,         # Décommenter si vous préférez 8-bit au lieu de 4-bit
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


def free_memory():
    """Libère la mémoire GPU et RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def load_and_prepare_data(csv_path, test_size=0.2, random_state=42, max_samples=None):
    """Charge et prépare le dataset (version allégée)."""
    print(f"\n📁 Chargement du dataset : {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Nombre total de contrats : {len(df)}")
    
    # Nettoyer
    df = df.dropna(subset=['code', 'label_encoded'])
    df['label_encoded'] = df['label_encoded'].astype(int)
    
    # ⚠️ LIMITER le nombre d'échantillons
    if max_samples and len(df) > max_samples:
        print(f"\n   ⚠️  LIMITATION à {max_samples} contrats pour économiser RAM")
        df = df.sample(n=max_samples, random_state=random_state)
        df = df.reset_index(drop=True)
    
    print(f"   Après nettoyage : {len(df)} contrats")
    print(f"\n   Distribution des labels :")
    for label, count in df['label_encoded'].value_counts().sort_index().items():
        print(f"      {label} ({LABELS_8[label]}): {count}")
    
    # Créer les prompts
    print("\n   Création des prompts...")
    df['text'] = df.apply(
        lambda row: create_prompt(row['code'], row['label_encoded']),
        axis=1
    )
    
    # Split train/test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label_encoded']
    )
    
    print(f"\n   📊 Split des données :")
    print(f"      Train : {len(train_df)} contrats")
    print(f"      Test  : {len(test_df)} contrats")
    
    # Convertir en Dataset
    train_dataset = Dataset.from_pandas(train_df[['text']])
    test_dataset = Dataset.from_pandas(test_df[['text']])
    
    # Libérer mémoire
    del df, train_df
    free_memory()
    
    return train_dataset, test_dataset, test_df


def load_model_and_tokenizer(model_name, use_4bit=True):
    """Charge le modèle et tokenizer (version optimisée)."""
    print(f"\n🤖 Chargement du modèle : {model_name}")
    
    if use_4bit:
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
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # ⚠️ IMPORTANT
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,  # ⚠️ IMPORTANT
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print(f"   ✅ Modèle chargé")
    
    # Libérer mémoire
    free_memory()
    
    return model, tokenizer


def setup_lora(model, config):
    """Configure LoRA (version optimisée)."""
    print(f"\n⚙️  Configuration de LoRA")
    
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
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"   Paramètres entraînables : {trainable_params:,}")
    print(f"   Tous les paramètres : {all_params:,}")
    print(f"   Pourcentage entraînable : {100 * trainable_params / all_params:.2f}%")
    
    # Libérer mémoire
    free_memory()
    
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
    """Entraîne le modèle (version optimisée)."""
    print(f"\n🏋️  Début de l'entraînement")
    
    # Tokenizer les datasets
    print("   Tokenization des données...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    
    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test"
    )
    
    # Libérer mémoire
    free_memory()
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Arguments d'entraînement OPTIMISÉS
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
        save_total_limit=2,  # ⚠️ Garde seulement 2 checkpoints
        gradient_checkpointing=True,  # ⚠️ IMPORTANT pour économiser RAM
        optim="paged_adamw_8bit",  # ⚠️ Optimiseur 8-bit
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )
    
    print("   🚀 Lancement de l'entraînement...")
    print(f"   ⚠️  Cela peut prendre 1-3 heures (modèle léger)")
    
    trainer.train()
    
    print(f"\n   ✅ Entraînement terminé !")
    
    # Sauvegarder
    final_model_path = os.path.join(config["output_dir"], "final_model")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"   💾 Modèle sauvegardé dans : {final_model_path}")
    
    # Libérer mémoire
    free_memory()
    
    return trainer


def evaluate_model(model, tokenizer, test_df, max_length=1024):
    """Évalue le modèle."""
    print(f"\n📊 Évaluation du modèle")
    
    predictions = []
    true_labels = []
    
    model.eval()
    
    # Limiter l'évaluation pour économiser du temps
    eval_samples = min(100, len(test_df))
    print(f"   Évaluation sur {eval_samples} échantillons")
    
    for idx, row in test_df.head(eval_samples).iterrows():
        prompt = create_prompt(row['code'])
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
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
        
        if (idx + 1) % 10 == 0:
            print(f"   Progression : {idx + 1}/{eval_samples}")
            free_memory()
    
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(predictions) * 100
    
    print(f"\n   ✅ Précision sur {eval_samples} échantillons : {accuracy:.2f}%")
    
    return predictions, true_labels, accuracy


def save_config(config, output_dir):
    """Sauvegarde la configuration."""
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Configuration sauvegardée : {config_path}")


if __name__ == "__main__":
    print("="*60)
    print("FINE-TUNING DE LLAMA AVEC LoRA (VERSION LOW RAM)")
    print("="*60)
    
    # Vérifier CUDA
    if torch.cuda.is_available():
        print(f"\n✅ GPU disponible : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Mémoire GPU : {mem:.2f} GB")
        
        if mem < 4:
            print("\n   ⚠️  ATTENTION : Moins de 4GB de VRAM")
            print("   → Utilisez TinyLlama (déjà configuré)")
            print("   → Réduisez max_samples à 200 si problème")
    else:
        print("\n⚠️  Pas de GPU détecté - entraînement très lent")
    
    print("\n" + "="*60)
    print("CONFIGURATION LOW RAM ACTIVE")
    print("="*60)
    print(f"Modèle : {CONFIG['model_name']}")
    print(f"Max samples : {CONFIG['max_samples']}")
    print(f"Batch size : {CONFIG['per_device_train_batch_size']}")
    print(f"Max length : {CONFIG['max_length']}")
    print(f"Époques : {CONFIG['num_train_epochs']}")
    print("="*60)
    
    response = input("\nContinuer ? (y/n): ")
    if response.lower() != 'y':
        print("Annulé")
        exit(0)
    
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
    
    # 3. Configurer LoRA
    model = setup_lora(model, CONFIG)
    
    # 4. Sauvegarder config
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    save_config(CONFIG, CONFIG["output_dir"])
    
    # 5. Entraîner
    trainer = train_model(model, tokenizer, train_dataset, test_dataset, CONFIG)
    
    # 6. Évaluer
    predictions, true_labels, accuracy = evaluate_model(
        model, tokenizer, test_df, CONFIG["max_length"]
    )
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ !")
    print("="*60)
    print(f"\n✅ Modèle sauvegardé dans : {CONFIG['output_dir']}/final_model")
    print(f"✅ Précision : {accuracy:.2f}%")
