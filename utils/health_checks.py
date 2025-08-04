
import nltk
import torch
import requests
import psutil
import sys


def check_transfeatex():
    import os

    use_vpn = os.environ.get('TRANSFEATEX_USE_VPN', 'true').lower() == 'true'

    if use_vpn:
        endpoint = 'http://10.4.63.10:3004/extract-features'
        test_input = {
            "text": [
                {"id": "test1", "text": "the app video camera video call chat text video send message I love"}
            ]
        }
    else:
        endpoint = 'http://gessi-chatbots.essi.upc.edu:3004/extract-features-aux'
        test_input = {"text": "the app video camera video call chat text video send message I love"}

    try:
        response = requests.post(endpoint, json=test_input, timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "endpoint": endpoint, "endpoint_type": "VPN" if use_vpn else "Original"}
        else:
            return {"status": "unhealthy", "error": f"Unexpected status code: {response.status_code}",
                    "endpoint": endpoint, "endpoint_type": "VPN" if use_vpn else "Original"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "endpoint": endpoint,
                "endpoint_type": "VPN" if use_vpn else "Original"}


def check_neo4j(neo4j_conn):
    try:
        with neo4j_conn.driver.session(database=neo4j_conn.database) as session:
            if session.run("RETURN 1 as test").single()["test"] == 1:
                return {"status": "healthy", "database": neo4j_conn.database, "uri": neo4j_conn.uri}
    except:
        pass
    return {"status": "unhealthy", "error": "Cannot connect to Neo4j", "database": neo4j_conn.database, "uri": neo4j_conn.uri}


def check_tfrex_model(feature_extractor):
    try:
        if feature_extractor.model and feature_extractor.tokenizer:
            test = feature_extractor.extract_features(["test app notification"])
            is_valid = test is not None and hasattr(test, '__len__') and len(test) > 0
            return {"status": "healthy", "model_name": feature_extractor.model_name, "test_result": is_valid}
        else:
            raise ValueError("Model or tokenizer not loaded")
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "model_name": getattr(feature_extractor, "model_name", "unknown")}


def check_embedding_model(feature_extractor):
    try:
        if feature_extractor.embedding_model:
            test_embeddings = feature_extractor.get_embeddings(["test text"])
            is_valid = test_embeddings is not None and hasattr(test_embeddings, 'shape') and test_embeddings.shape[0] > 0
            return {"status": "healthy", "model_name": "all-MiniLM-L6-v2", "test_result": is_valid}
        else:
            raise ValueError("Embedding model not loaded")
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "model_name": "all-MiniLM-L6-v2"}


def check_nltk_data():
    try:
        required = ['punkt', 'stopwords', 'wordnet']
        for item in required:
            try:
                nltk.data.find(f'corpora/{item}' if item != 'punkt' else f'tokenizers/{item}')
            except LookupError:
                nltk.download(item)
        return {"status": "healthy", "data_available": required}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_ollama(config):
    try:
        model_name = config['OLLAMA_MODEL']
        base_url = config['OLLAMA_BASE_URL']

        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code != 200:
            return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}

        models = response.json().get("models", [])
        if not any(model.get("name") == model_name for model in models):
            return {"status": "unhealthy", "error": f"Model '{model_name}' not found in Ollama"}

        chat_resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say hello"}]
            },
            timeout=5
        )

        if chat_resp.status_code == 200:
            return {"status": "healthy", "model": model_name}
        else:
            return {"status": "unhealthy", "error": f"Chat failed with status {chat_resp.status_code}", "model": model_name}

    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "model": config.get('OLLAMA_MODEL')}
