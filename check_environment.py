"""
check_environment.py
Script de vérification de l'environnement avant le fine-tuning
"""

import sys
import subprocess

def check_python_version():
    """Vérifie la version de Python."""
    print("🐍 Vérification de Python...")
    version = sys.version_info
    print(f"   Version Python : {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ requis")
        return False
    else:
        print("   ✅ Version Python OK")
        return True


def check_cuda():
    """Vérifie la disponibilité de CUDA."""
    print("\n🔥 Vérification de CUDA...")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA disponible")
            print(f"   GPU : {torch.cuda.get_device_name(0)}")
            print(f"   Version CUDA : {torch.version.cuda}")
            print(f"   Nombre de GPUs : {torch.cuda.device_count()}")
            
            # Vérifier la mémoire GPU
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1e9
                print(f"   GPU {i} - Mémoire totale : {total_memory:.2f} GB")
                
                if total_memory < 8:
                    print(f"   ⚠️  GPU {i} a moins de 8GB, risque de mémoire insuffisante")
                    print(f"      → Réduisez batch_size et max_length")
            
            return True
        else:
            print("   ⚠️  CUDA non disponible - l'entraînement sera sur CPU (très lent)")
            print("      → Installez PyTorch avec support CUDA :")
            print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
            
    except ImportError:
        print("   ❌ PyTorch non installé")
        return False


def check_packages():
    """Vérifie l'installation des packages requis."""
    print("\n📦 Vérification des packages...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'peft': 'PEFT (LoRA)',
        'bitsandbytes': 'BitsAndBytes',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'tqdm': 'TQDM',
    }
    
    all_ok = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} non installé")
            all_ok = False
    
    if not all_ok:
        print("\n   → Installez les packages manquants avec :")
        print("   pip install -r requirements.txt")
    
    return all_ok


def check_dataset():
    """Vérifie la présence du dataset."""
    print("\n📊 Vérification du dataset...")
    
    import os
    import pandas as pd
    
    dataset_path = "archive/SC_Vuln_8label.csv"
    
    if not os.path.exists(dataset_path):
        print(f"   ❌ Dataset non trouvé : {dataset_path}")
        print("      → Assurez-vous que le fichier existe")
        return False
    
    print(f"   ✅ Dataset trouvé : {dataset_path}")
    
    # Charger et vérifier le dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"   Nombre de lignes : {len(df)}")
        
        # Vérifier les colonnes requises
        required_cols = ['code', 'label_encoded']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"   ❌ Colonnes manquantes : {missing_cols}")
            return False
        
        print(f"   ✅ Colonnes OK : {list(df.columns)}")
        
        # Distribution des labels
        print("\n   Distribution des labels :")
        for label, count in df['label_encoded'].value_counts().sort_index().items():
            print(f"      Label {int(label)} : {count} contrats")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur lors du chargement : {e}")
        return False


def check_disk_space():
    """Vérifie l'espace disque disponible."""
    print("\n💾 Vérification de l'espace disque...")
    
    import shutil
    
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (2**30)
    
    print(f"   Espace libre : {free_gb:.2f} GB")
    
    if free_gb < 20:
        print("   ⚠️  Moins de 20GB disponibles")
        print("      → Le fine-tuning nécessite ~20GB pour le modèle et les checkpoints")
        return False
    else:
        print("   ✅ Espace disque suffisant")
        return True


def check_memory():
    """Vérifie la RAM disponible."""
    print("\n🧠 Vérification de la RAM...")
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        total_gb = mem.total / (2**30)
        available_gb = mem.available / (2**30)
        
        print(f"   RAM totale : {total_gb:.2f} GB")
        print(f"   RAM disponible : {available_gb:.2f} GB")
        
        if available_gb < 8:
            print("   ⚠️  Moins de 8GB de RAM disponible")
            print("      → Fermez les applications inutiles")
            return False
        else:
            print("   ✅ RAM suffisante")
            return True
            
    except ImportError:
        print("   ⚠️  psutil non installé (pip install psutil)")
        print("   Vérification de la RAM ignorée")
        return True


def estimate_training_time():
    """Estime le temps d'entraînement."""
    print("\n⏱️  Estimation du temps d'entraînement...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            
            # Estimations approximatives
            estimates = {
                "RTX 4090": "1-2 heures",
                "RTX 3090": "2-4 heures",
                "RTX 3080": "3-5 heures",
                "RTX 3060": "4-8 heures",
                "RTX 2080": "5-10 heures",
                "default": "4-8 heures (selon le GPU)"
            }
            
            time_estimate = "inconnue"
            for gpu_model, estimate in estimates.items():
                if gpu_model in gpu_name:
                    time_estimate = estimate
                    break
            
            if time_estimate == "inconnue":
                time_estimate = estimates["default"]
            
            print(f"   GPU : {gpu_name}")
            print(f"   Temps estimé (3 époques) : {time_estimate}")
            
        else:
            print("   CPU uniquement : plusieurs jours (non recommandé)")
            
    except Exception as e:
        print(f"   Impossible d'estimer : {e}")


def run_test_inference():
    """Test rapide d'inférence."""
    print("\n🧪 Test rapide d'inférence...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("   Chargement d'un petit modèle de test...")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("   ✅ Chargement OK")
        
        # Test simple
        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)
        
        print("   ✅ Génération OK")
        print("   Le système est prêt pour le fine-tuning !")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur lors du test : {e}")
        print("   → Vérifiez votre installation")
        return False


def main():
    """Fonction principale."""
    print("="*60)
    print("VÉRIFICATION DE L'ENVIRONNEMENT")
    print("Fine-tuning LLaMA avec LoRA")
    print("="*60)
    
    checks = {
        "Python": check_python_version(),
        "CUDA": check_cuda(),
        "Packages": check_packages(),
        "Dataset": check_dataset(),
        "Espace disque": check_disk_space(),
        "RAM": check_memory(),
    }
    
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
    
    estimate_training_time()
    
    all_ok = all(checks.values())
    
    if all_ok:
        print("\n" + "="*60)
        print("✅ ENVIRONNEMENT PRÊT POUR LE FINE-TUNING")
        print("="*60)
        print("\nVous pouvez lancer l'entraînement avec :")
        print("python fine_tune_llama_lora.py")
        
        # Test optionnel
        print("\n" + "="*60)
        response = input("\nVoulez-vous faire un test d'inférence rapide ? (y/n): ")
        if response.lower() == 'y':
            run_test_inference()
        
    else:
        print("\n" + "="*60)
        print("⚠️  PROBLÈMES DÉTECTÉS")
        print("="*60)
        print("\nCorrigez les problèmes avant de lancer l'entraînement.")
        print("Consultez le README.md pour plus d'informations.")


if __name__ == "__main__":
    main()
