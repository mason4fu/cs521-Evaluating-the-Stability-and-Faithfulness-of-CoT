"""Model runner for Llama-4-Maverick supporting vLLM and llama.cpp via SSH"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import subprocess
import json
import re
from typing import Dict, List, Optional, Tuple

import config


class ModelRunner:
    """Runner for Llama-4-Maverick model via vLLM or llama.cpp on VM"""
    
    def __init__(
        self,
        runtime: str = None,
        ssh_host: str = None,
        ssh_user: str = None,
        model_path: str = None,
        use_local: bool = False
    ):
        """
        Args:
            runtime: "vllm" or "llamacpp"
            ssh_host: VM host (e.g., "149.165.151.46")
            ssh_user: VM user (e.g., "exouser")
            model_path: Model identifier (HuggingFace model name)
            use_local: If True, run locally instead of via SSH
        """
        self.runtime = runtime or config.MODEL_RUNTIME
        self.ssh_host = ssh_host or config.VM_HOST
        self.ssh_user = ssh_user or config.VM_USER
        # Use model name from config (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        self.model_path = model_path or config.MODEL_NAME
        self.use_local = use_local or (ssh_host is None)
        
        print(f"ModelRunner initialized: runtime={self.runtime}, use_local={self.use_local}")
        print(f"  Model: {self.model_path}")
        if not self.use_local:
            print(f"  SSH: {self.ssh_user}@{self.ssh_host}")
    
    def _ssh_command(self, command: str) -> str:
        """Execute command on VM via SSH"""
        full_command = f'ssh {self.ssh_user}@{self.ssh_host} "{command}"'
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"SSH command failed: {result.stderr}")
        return result.stdout.strip()
    
    def _local_command(self, command: str) -> str:
        """Execute command locally"""
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")
        return result.stdout.strip()
    
    def generate(
        self,
        prompt: str,
        temperature: float = config.TEMPERATURE,
        top_p: float = config.NUCLEUS_P,
        max_tokens: int = config.MAX_TOKENS,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling p
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
        
        Returns:
            Generated text
        """
        if self.runtime == "vllm":
            return self._generate_vllm(prompt, temperature, top_p, max_tokens, stop)
        elif self.runtime == "llamacpp":
            return self._generate_llamacpp(prompt, temperature, top_p, max_tokens, stop)
        else:
            raise ValueError(f"Unknown runtime: {self.runtime}")
    
    def _generate_vllm_api(self, api_url: str, prompt: str, temperature: float, top_p: float, max_tokens: int, stop: List[str]) -> str:
        """Generate using vLLM API server (fast - model already loaded)"""
        import requests
        
        # Truncate prompt if too long (vLLM server has max_model_len=2048)
        # Leave room for generation tokens
        max_prompt_tokens = 1800  # Conservative limit
        # Rough estimate: 1 token ≈ 4 characters
        max_prompt_chars = max_prompt_tokens * 4
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars]
        
        # Use OpenAI-compatible API endpoint
        url = f"{api_url}/v1/completions"
        
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stop": stop if stop else None
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                # Bad request - might be prompt too long or invalid format
                error_detail = ""
                try:
                    error_detail = e.response.json().get("error", {}).get("message", "")
                except:
                    pass
                raise RuntimeError(f"vLLM API request failed (400 Bad Request): {error_detail or 'Invalid request format or prompt too long'}")
            raise RuntimeError(f"vLLM API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"vLLM API request failed: {e}")
    
    def _generate_vllm(self, prompt: str, temperature: float, top_p: float, max_tokens: int, stop: List[str]) -> str:
        """Generate using vLLM - either via API server or by loading model directly"""
        # Check if we should use API server
        api_url = config.VLLM_API_URL
        if api_url and not api_url.startswith("None"):
            return self._generate_vllm_api(api_url, prompt, temperature, top_p, max_tokens, stop)
        
        # Fallback to direct model loading (slow)
        import base64
        
        # Create a temporary Python script for vLLM
        script_content = f"""import sys
import os
import json
from vllm import LLM, SamplingParams

# Set environment variables for volume storage
os.environ['HF_HOME'] = '/media/volume/cot-llm-storage/hf-cache'
os.environ['TMPDIR'] = '/media/volume/cot-llm-storage/tmp'
os.environ['TMP'] = '/media/volume/cot-llm-storage/tmp'
os.environ['TEMP'] = '/media/volume/cot-llm-storage/tmp'
hf_token = os.getenv('HF_TOKEN', '')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

# Load model (loads fresh each time - slow but works)
# TODO: For production, use persistent vLLM server
model_id = {json.dumps(self.model_path)}
try:
    model = LLM(
        model=model_id,
        gpu_memory_utilization=0.70,  # Reduced to allow for other processes
        max_model_len=2048,
        dtype='float16',
        download_dir='/media/volume/cot-llm-storage/models'
    )
    
    # Generate
    stop_list = {json.dumps(stop) if stop else "None"}
    sampling_params = SamplingParams(
        temperature={temperature},
        top_p={top_p},
        max_tokens={max_tokens},
        stop=stop_list if stop_list else None
    )
    
    prompt_text = {json.dumps(prompt)}
    outputs = model.generate([prompt_text], sampling_params)
    
    result = outputs[0].outputs[0].text
    print(json.dumps({{"text": result}}))
except Exception as e:
    import traceback
    error_msg = str(e) + "\\n" + traceback.format_exc()
    print(json.dumps({{"error": error_msg}}))
    sys.exit(1)
"""
        
        if self.use_local:
            # For local execution, write script to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_file = f.name
            
            try:
                env_vars = f'HF_HOME=/media/volume/cot-llm-storage/hf-cache TMPDIR=/media/volume/cot-llm-storage/tmp HF_TOKEN={os.getenv("HF_TOKEN", "")}'
                result = self._local_command(f'{env_vars} {config.VM_PYTHON_PATH} {script_file}')
            finally:
                os.unlink(script_file)
        else:
            # Write script to VM using base64 to avoid escaping issues
            script_b64 = base64.b64encode(script_content.encode()).decode()
            script_file = f"{config.VM_HOME}/tmp_vllm_gen_{os.getpid()}.py"
            
            # Decode and write script on VM
            self._ssh_command(f'echo {script_b64} | base64 -d > {script_file}')
            
            # Run with environment variables
            env_vars = 'HF_HOME=/media/volume/cot-llm-storage/hf-cache TMPDIR=/media/volume/cot-llm-storage/tmp'
            hf_token = os.getenv("HF_TOKEN", "")
            if hf_token:
                env_vars += f' HF_TOKEN={hf_token}'
            
            result = self._ssh_command(f'{env_vars} {config.VM_PYTHON_PATH} {script_file}')
            
            # Clean up script file
            self._ssh_command(f'rm -f {script_file}')
        
        try:
            output = json.loads(result)
            if "error" in output:
                raise RuntimeError(f"vLLM generation error: {output['error']}")
            return output["text"]
        except json.JSONDecodeError:
            # Fallback: if JSON parsing fails, return raw output
            return result
    
    def _generate_llamacpp(self, prompt: str, temperature: float, top_p: float, max_tokens: int, stop: List[str]) -> str:
        """Generate using llama.cpp"""
        # For llama.cpp, we'd use llama-cli or similar
        # This is a placeholder - actual implementation depends on llama.cpp setup
        raise NotImplementedError("llama.cpp runtime not yet implemented. Use vLLM for now.")
    
    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = config.TEMPERATURE,
        top_p: float = config.NUCLEUS_P,
        max_tokens: int = config.MAX_TOKENS,
        stop: Optional[List[str]] = None
    ) -> List[str]:
        """Generate text for multiple prompts (batched)"""
        if self.runtime == "vllm":
            return self._batch_generate_vllm(prompts, temperature, top_p, max_tokens, stop)
        else:
            # Fallback to sequential generation
            return [self.generate(p, temperature, top_p, max_tokens, stop) for p in prompts]
    
    def _batch_generate_vllm(self, prompts: List[str], temperature: float, top_p: float, max_tokens: int, stop: List[str]) -> List[str]:
        """Batch generate using vLLM - prefer API server if available"""
        # Check if we should use API server (faster - model already loaded)
        api_url = config.VLLM_API_URL
        if api_url and not api_url.startswith("None") and api_url != "":
            return self._batch_generate_vllm_api(api_url, prompts, temperature, top_p, max_tokens, stop)
        
        # Fallback to loading model directly (slow)
        script = f"""
import sys
import json
from vllm import LLM, SamplingParams

model = LLM(model="{self.model_path}")

sampling_params = SamplingParams(
    temperature={temperature},
    top_p={top_p},
    max_tokens={max_tokens},
    stop={stop if stop else "None"}
)

prompts = {json.dumps(prompts)}
outputs = model.generate(prompts, sampling_params)

results = [o.outputs[0].text for o in outputs]
print(json.dumps(results))
"""
        
        if self.use_local:
            result = self._local_command(f'{config.VM_PYTHON_PATH} -c {json.dumps(script)}')
        else:
            script_file = f"{config.VM_HOME}/tmp_vllm_batch.py"
            self._ssh_command(f'echo {json.dumps(script)} > {script_file}')
            result = self._ssh_command(f'{config.VM_PYTHON_PATH} {script_file}')
        
        try:
            return json.loads(result)
        except:
            return result.split('\n')
    
    def _batch_generate_vllm_api(self, api_url: str, prompts: List[str], temperature: float, top_p: float, max_tokens: int, stop: List[str]) -> List[str]:
        """Batch generate using vLLM API server (fast - model already loaded)"""
        import requests
        
        # vLLM API supports batch requests - send all prompts in parallel
        # Use threading to send requests in parallel for better throughput
        import concurrent.futures
        
        def generate_single(prompt: str) -> str:
            url = f"{api_url}/v1/completions"
            payload = {
                "model": self.model_path,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stop": stop if stop else None
            }
            try:
                response = requests.post(url, json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["text"]
            except Exception as e:
                raise RuntimeError(f"vLLM API request failed: {e}")
        
        # Process in parallel (up to 8 concurrent requests)
        max_workers = min(8, len(prompts))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(generate_single, prompts))
        
        return results


# For local testing without SSH (using transformers library)
class LocalModelRunner:
    """Local model runner using transformers library (for testing)"""
    
    def __init__(self, model_name: str = config.MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        print(f"LocalModelRunner will load: {model_name}")
        print("Note: Model will be loaded lazily on first generation")
    
    def _load_model(self):
        """Lazy load model"""
        if self.model is not None:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading {self.model_name} locally...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("✅ Model loaded")
    
    def generate(
        self,
        prompt: str,
        temperature: float = config.TEMPERATURE,
        top_p: float = config.NUCLEUS_P,
        max_tokens: int = config.MAX_TOKENS,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate text locally"""
        import torch
        
        self._load_model()
        
        # Use chat template if available (for instruct models like Ministral)
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            # Format as a chat message for instruct models
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in text:
                    text = text.split(stop_seq)[0]
        
        return text
    
    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = config.TEMPERATURE,
        top_p: float = config.NUCLEUS_P,
        max_tokens: int = config.MAX_TOKENS,
        stop: Optional[List[str]] = None
    ) -> List[str]:
        """Batch generate locally"""
        return [self.generate(p, temperature, top_p, max_tokens, stop) for p in prompts]

