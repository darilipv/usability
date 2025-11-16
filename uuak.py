#!/usr/bin/env python

import random
import os
import ollama
import nltk
from data_storage import JSONDataStorage


class Modifier:

    def __init__(self, high, low, min=-2, max=2):
        self._high = high
        self._low  = low

    def QUANT_STR(self, value):
        value = abs(value)
        if value == 0:
            return ""
        elif value == 1:
            return "somewhat "
        elif value == 2:
            return "very "
        else:
            raise ValueError("Value must be -2, -1, 0, 1, or 2.")

    def quant(self, value):
        if value > 0:
            return f"{self.QUANT_STR(value)}{self._high}"
        elif value < 0:
            return f"{self.QUANT_STR(value)}{self._low}"
        else:
            return ""

    def random_value(self):
        return random.choice([-2, -1, 1, 2])

    def random_quant_str(self):
        return self.quant(self.random_value())


MODIFIERS = [
    Modifier('long', 'short'),
    Modifier('factual', 'emotional'),
    Modifier('advanced', 'simple'),
    Modifier('expert-oriented', 'layman-oriented'),
    Modifier('formal', 'informal'),
    Modifier('creative', 'straightforward'),
    Modifier('enthusiastic', 'reserved'),
]

def pick_random_combination(n=3):
    sample = random.sample(MODIFIERS, n)
    return ", ".join([m.random_quant_str() for m in sample])



def pull_model(model):
    # os.system(f"ollama pull {model}")
    pass


def get_available_ollama_models(debug=False):
    """
    Get list of all available Ollama models.
    Tries Python API first, falls back to command line.
    
    Args:
        debug: If True, print debug information about the API response
    
    Returns:
        List of model names (strings)
    """
    models_list = []
    
    # Method 1: Try Python API
    try:
        response = ollama.list()
        
        if debug:
            print(f"DEBUG: Response type: {type(response)}")
            print(f"DEBUG: Response content: {response}")
        
        # Handle different possible response structures
        # Case 1: Direct list of models
        if isinstance(response, list):
            for model in response:
                if isinstance(model, dict):
                    name = model.get('name') or model.get('model') or model.get('id')
                    if name:
                        models_list.append(name)
                elif isinstance(model, str):
                    models_list.append(model)
        
        # Case 2: Dictionary with 'models' key
        elif isinstance(response, dict):
            if 'models' in response:
                models_data = response['models']
                if isinstance(models_data, list):
                    for model in models_data:
                        if isinstance(model, dict):
                            name = model.get('name') or model.get('model') or model.get('id')
                            if name:
                                models_list.append(name)
                        elif isinstance(model, str):
                            models_list.append(model)
            else:
                # Case 3: Dictionary without 'models' key - check all values
                for key, value in response.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                name = item.get('name') or item.get('model') or item.get('id')
                                if name:
                                    models_list.append(name)
                            elif isinstance(item, str):
                                models_list.append(item)
        
        if models_list:
            return list(dict.fromkeys(models_list))
            
    except Exception as e:
        if debug:
            print(f"DEBUG: Python API failed: {e}")
    
    # Method 2: Fallback to command line
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # Skip header line (usually "NAME" or similar)
            for line in lines[1:]:
                if line.strip():
                    # Extract model name (first column)
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        # Skip header-like entries
                        if model_name.lower() not in ['name', 'model', '---']:
                            models_list.append(model_name)
            
            if models_list:
                return list(dict.fromkeys(models_list))
    except FileNotFoundError:
        if debug:
            print("DEBUG: 'ollama' command not found in PATH")
    except Exception as e:
        if debug:
            print(f"DEBUG: Command line fallback failed: {e}")
    
    return []


class Agent:

    def __init__(self):
        pass

    def name(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def ask(self, prompt):
        raise NotImplementedError("Subclasses should implement this method.")


class OllamaAgent(Agent):

    def __init__(self, model, system_prompt=None):
        super().__init__()
        self.model         = model
        self.system_prompt = system_prompt
        pull_model(model)

    def name(self):
        return f"ollama/{self.model}"

    def ask(self, prompt):
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": full_prompt}])
        return response['message']


class SpecializedQuery:

    def __init__(self, tf, agent, base_prompt, combination):
        self.tf           = tf
        self.agent        = agent
        self.base_prompt  = base_prompt
        self.combination  = combination

    def full_prompt(self):
        prompt  = self.base_prompt
        prompt += " Please answer in a {} manner.".format(self.combination)
        return prompt

    def run(self):
        prompt    = self.full_prompt()
        response  = self.agent.ask(prompt)
        return response


class Database:

    def _ensure_exists(self):
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)

    def __init__(self, db_dir):
        self._directory = db_dir
        self._ensure_exists()



def create_sia():
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download(['vader_lexicon'],quiet=False)
    sia = SentimentIntensityAnalyzer()
    return sia

class TestFramework:

    def __init__(self, system_prompt="You are a helpful assistant.", data_storage=None):
        self._agents        = list()
        self._system_prompt = system_prompt
        self._sia           = create_sia()
        self._data_storage  = data_storage or JSONDataStorage()

    def with_agent(self, agent):
        self._agents.append(agent)
        return self

    def with_ollama_agent(self, model):
        agent = OllamaAgent(model, self._system_prompt)
        self._agents.append(agent)
        return self
    
    def run1(self, prompt):
        results = dict()
        for agent in self._agents:
            response = agent.ask(prompt)
            results[agent] = response.content
        return results

    def runN(self, n, prompt):
        results = dict()
        for i in range(n):
            for agent in self._agents:
                response = agent.ask(prompt)
                if agent not in results:
                    results[agent] = list()
                results[agent].append(response.content)
        return results

    def analyze_sentiment(self, text):
        sentiment = self._sia.polarity_scores(text)
        return sentiment

    def test(self, base_prompt, n_runs=3, save_data=True):
        """
        Run a test with a base prompt and random style combination.
        
        Args:
            base_prompt: The base prompt to test
            n_runs: Number of times to run the test with the same style combination
            save_data: Whether to save results to storage
        """
        comb    = pick_random_combination()
        prompt  = base_prompt
        prompt += " Please answer in a {} manner.".format(comb)
        print(f"Prompt: {prompt}\n")
        results = self.runN(n_runs, prompt)
        
        for agent, response_list in results.items():
            for resp in response_list:
                print(f"Agent: {agent.name()}\nResponse: {resp}\n")
                sentiment = self.analyze_sentiment(resp)
                print(f"Sentiment: {sentiment}\n")
                
                # Save test result
                if save_data:
                    test_data = {
                        'base_prompt': base_prompt,
                        'full_prompt': prompt,
                        'style_combination': comb,
                        'agent_name': agent.name(),
                        'response': resp,
                        'sentiment': sentiment
                    }
                    self._data_storage.save_test_result(test_data)
        
        print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run prompt stability tests with Ollama models'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to test (e.g., --models qwen2.5:0.5b qwen2.5:3b). If not specified, uses all available models.'
    )
    parser.add_argument(
        '--use-all',
        action='store_true',
        help='Use all available Ollama models automatically'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available Ollama models and exit'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = get_available_ollama_models(debug=args.debug)
        if models:
            print("Available Ollama models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No models found or could not connect to Ollama.")
            if args.debug:
                print("\nTrying with debug output...")
                get_available_ollama_models(debug=True)
        exit(0)
    
    # Determine which models to use
    if args.models:
        models_to_use = args.models
    elif args.use_all:
        models_to_use = get_available_ollama_models(debug=args.debug)
        if not models_to_use:
            print("Warning: No models found. Using default models.")
            if args.debug:
                print("Running debug mode to see API response structure...")
                get_available_ollama_models(debug=True)
            models_to_use = ["qwen2.5:0.5b", "qwen2.5:3b"]
    else:
        # Default: use all available models, or fallback to hardcoded ones
        models_to_use = get_available_ollama_models(debug=args.debug)
        if not models_to_use:
            print("No models found. Using default models.")
            if args.debug:
                print("Running debug mode to see API response structure...")
                get_available_ollama_models(debug=True)
            models_to_use = ["qwen2.5:0.5b", "qwen2.5:3b"]
    
    print(f"Testing with {len(models_to_use)} model(s): {', '.join(models_to_use)}")
    print()
    
    # Create test framework
    tf = TestFramework(system_prompt="You are a helpful assistant.")
    
    # Add all models as agents
    for model in models_to_use:
        tf.with_ollama_agent(model)
    
    # Run tests
    tf.test("Please intoduce yourself.")
    tf.test("Please tell me something about yourself.")
    tf.test("How do I get to the nearest airport?") # Trick question - check for hallucinations
    tf.test("What is the capital of France?")
    tf.test("Explain the theory of relativity.")
    tf.test("How does a blockchain work?")
    tf.test("What is a transformer model?")
    tf.test("Describe the process of photosynthesis.")
    tf.test("Who was Leonardo da Vinci?")
