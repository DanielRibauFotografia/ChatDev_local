# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import argparse
import logging
import os
import sys

from camel.typing import ModelType

root = os.path.dirname(__file__)
sys.path.append(root)

from chatdev.chat_chain import ChatChain

try:
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message import FunctionCall

    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version
    print(
        "Warning: Your OpenAI version is outdated. \n "
        "Please update as specified in requirement.txt. \n "
        "The old API interface is deprecated and will no longer be supported.")


def get_config(company):
    """
    return configuration json files for ChatChain
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        company: customized configuration name under CompanyConfig/

    Returns:
        path to three configuration jsons: [config_path, config_phase_path, config_role_path]
    """
    config_dir = os.path.join(root, "CompanyConfig", company)
    default_config_dir = os.path.join(root, "CompanyConfig", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json",
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)


parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--config', type=str, default="Default",
                    help="Name of config, which is used to load configuration under CompanyConfig/")
parser.add_argument('--org', type=str, default="DefaultOrganization",
                    help="Name of organization, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--task', type=str, default="Develop a basic Gomoku game.",
                    help="Prompt of software")
parser.add_argument('--name', type=str, default="Gomoku",
                    help="Name of software, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--model', type=str, default="GPT_3_5_TURBO",
                    help="Model type, choose from {'GPT_3_5_TURBO', 'GPT_4', 'GPT_4_TURBO', 'GPT_4O', 'GPT_4O_MINI', 'HUGGINGFACE', 'LLAMA_CPP', 'OLLAMA', 'LOCALAI'}")
parser.add_argument('--path', type=str, default="",
                    help="Your file directory, ChatDev will build upon your software in the Incremental mode")

# Local model configuration arguments
parser.add_argument('--model_name', type=str, default="",
                    help="Name of the local model (for HuggingFace, Ollama, LocalAI)")
parser.add_argument('--model_path', type=str, default="",
                    help="Path to local model file (for llama.cpp)")
parser.add_argument('--base_url', type=str, default="",
                    help="Base URL for API-based local models (Ollama: http://localhost:11434, LocalAI: http://localhost:8080)")
parser.add_argument('--device', type=str, default="auto",
                    help="Device for local model inference (auto, cpu, cuda, mps)")
parser.add_argument('--max_tokens', type=int, default=512,
                    help="Maximum tokens to generate")
parser.add_argument('--temperature', type=float, default=0.7,
                    help="Temperature for text generation")
parser.add_argument('--top_p', type=float, default=0.9,
                    help="Top-p value for text generation")
args = parser.parse_args()

# Start ChatDev

# ----------------------------------------
#          Init ChatChain
# ----------------------------------------
config_path, config_phase_path, config_role_path = get_config(args.config)
args2type = {'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
             'GPT_4': ModelType.GPT_4,
            #  'GPT_4_32K': ModelType.GPT_4_32k,
             'GPT_4_TURBO': ModelType.GPT_4_TURBO,
            #  'GPT_4_TURBO_V': ModelType.GPT_4_TURBO_V
            'GPT_4O': ModelType.GPT_4O,
            'GPT_4O_MINI': ModelType.GPT_4O_MINI,
            'HUGGINGFACE': ModelType.HUGGINGFACE,
            'LLAMA_CPP': ModelType.LLAMA_CPP,
            'OLLAMA': ModelType.OLLAMA,
            'LOCALAI': ModelType.LOCALAI,
             }
if openai_new_api:
    args2type['GPT_3_5_TURBO'] = ModelType.GPT_3_5_TURBO_NEW

# Prepare model configuration for local models
model_config = {}
if args.model in ['HUGGINGFACE', 'LLAMA_CPP', 'OLLAMA', 'LOCALAI']:
    if args.model_name:
        model_config['model_name'] = args.model_name
    if args.model_path:
        model_config['model_path'] = args.model_path
    if args.base_url:
        model_config['base_url'] = args.base_url
    model_config['device'] = args.device
    model_config['max_tokens'] = args.max_tokens
    model_config['temperature'] = args.temperature
    model_config['top_p'] = args.top_p
    
    # Set defaults for specific backends
    if args.model == 'HUGGINGFACE' and not args.model_name:
        model_config['model_name'] = 'microsoft/DialoGPT-medium'
    elif args.model == 'OLLAMA':
        if not args.base_url:
            model_config['base_url'] = 'http://localhost:11434'
        if not args.model_name:
            model_config['model_name'] = 'llama2'
    elif args.model == 'LOCALAI':
        if not args.base_url:
            model_config['base_url'] = 'http://localhost:8080'
        if not args.model_name:
            model_config['model_name'] = 'gpt-3.5-turbo'

chat_chain = ChatChain(config_path=config_path,
                       config_phase_path=config_phase_path,
                       config_role_path=config_role_path,
                       task_prompt=args.task,
                       project_name=args.name,
                       org_name=args.org,
                       model_type=args2type[args.model],
                       model_config=model_config,
                       code_path=args.path)

# ----------------------------------------
#          Init Log
# ----------------------------------------
logging.basicConfig(filename=chat_chain.log_filepath, level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8")

# ----------------------------------------
#          Pre Processing
# ----------------------------------------

chat_chain.pre_processing()

# ----------------------------------------
#          Personnel Recruitment
# ----------------------------------------

chat_chain.make_recruitment()

# ----------------------------------------
#          Chat Chain
# ----------------------------------------

chat_chain.execute_chain()

# ----------------------------------------
#          Post Processing
# ----------------------------------------

chat_chain.post_processing()
