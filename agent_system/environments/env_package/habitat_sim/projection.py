import torch
import random
import re
import json
from typing import List, Tuple


# Define the list of valid task types
VALID_TASK_TYPES = ["grounding", "segment", "3d-box", "detect", "class-detect"]

# Define the action list
ACTION_LIST = ["move_forward", "turn_left", "turn_right", "look_up", "look_down", "stop"]


def parse_json_response(text: str) -> dict:
    """
    Try to parse JSON from the model's response.
    Handles cases where JSON is embedded in other text.
    """
    if not isinstance(text, str):
        return {}
    
    # Try to find JSON block in the text
    # Look for content between { and }
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to parse the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    return {}


def extract_task_type(text: str) -> Tuple[str, bool]:
    """
    Extract task_type from the model's response.
    
    Returns:
        (task_type, is_valid): The extracted task type and whether it was valid
    """
    if not isinstance(text, str):
        return random.choice(VALID_TASK_TYPES), False
    
    text_lower = text.lower()
    
    # First, try to parse as JSON
    parsed = parse_json_response(text)
    if parsed and "task_type" in parsed:
        task_type = parsed["task_type"].lower().strip()
        if task_type in VALID_TASK_TYPES:
            return task_type, True
    
    # Fallback: look for "task_type": pattern in raw text
    task_type_match = re.search(r'"task_type"\s*:\s*"([^"]+)"', text_lower)
    if task_type_match:
        task_type = task_type_match.group(1).strip()
        if task_type in VALID_TASK_TYPES:
            return task_type, True
    
    # Last resort: check if any task type keyword appears in the text
    found_types = []
    for task_type in VALID_TASK_TYPES:
        if task_type in text_lower:
            found_types.append(task_type)
    
    if len(found_types) == 1:
        return found_types[0], True
    
    # If we can't determine the task type, return a random one
    return random.choice(VALID_TASK_TYPES), False


def extract_action(text: str) -> Tuple[int, bool]:
    """
    Extract action from the model's response.
    
    Returns:
        (action_index, is_valid): The action index and whether it was valid
    """
    if not isinstance(text, str):
        return random.randint(0, len(ACTION_LIST) - 1), False
    
    text_lower = text.lower()
    
    # First, try to parse as JSON
    parsed = parse_json_response(text_lower)
    if parsed and "action" in parsed:
        action = parsed["action"].lower().strip()
        if action in ACTION_LIST:
            return ACTION_LIST.index(action), True
    
    # Fallback: look for "action": pattern in raw text
    action_index = text_lower.find('"action":')
    if action_index != -1:
        action_text = text_lower[action_index:]
        for i, action in enumerate(ACTION_LIST):
            if action in action_text:
                return i, True
    
    # Last resort: check for action keywords anywhere
    found_actions = []
    for i, action in enumerate(ACTION_LIST):
        if action in text_lower:
            found_actions.append((i, action))
    
    if len(found_actions) == 1:
        return found_actions[0][0], True
    
    # If we can't determine the action, return a random one
    return random.randint(0, len(ACTION_LIST) - 1), False


def extract_task_prompt(text: str) -> Tuple[str, bool]:
    """
    Extract task_prompt from the model's response.
    
    Returns:
        (task_prompt, is_valid): The extracted task prompt and whether it was valid
    """
    if not isinstance(text, str):
        return "", False
    
    # First, try to parse as JSON (use original text to preserve case)
    parsed = parse_json_response(text)
    if parsed and "task_prompt" in parsed:
        task_prompt = parsed["task_prompt"].strip()
        if task_prompt:
            return task_prompt, True
    
    # Fallback: look for "task_prompt": pattern in raw text
    task_prompt_match = re.search(r'"task_prompt"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if task_prompt_match:
        task_prompt = task_prompt_match.group(1).strip()
        if task_prompt:
            return task_prompt, True
    
    # If we can't determine the task_prompt, return empty string
    return "", False


def habitat_projection(text_actions: List[str], env_name: str) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Process model outputs to extract task_type, task_prompt, and action.
    
    This function parses the model's JSON response to extract:
    1. pred_task_type: The task type the model predicts (grounding, segment, 3d-box)
    2. pred_task_prompt: The target object description extracted by the model
    3. action_index: The action to take (0-5 corresponding to action_list)
    4. valid: Whether the parsing was successful
    
    Args:
        text_actions: List of model output strings
        env_name: Environment name (should be "habitat")
    
    Returns:
        pred_task_types: List of predicted task types
        pred_task_prompts: List of predicted task prompts
        action_indices: List of action indices
        valids: List of validity flags (1 if task_type, task_prompt, and action were all valid, 0 otherwise)
    """
    if env_name != 'habitat':
        raise NotImplementedError("Action list not implemented for this env!")
    
    pred_task_types = []
    pred_task_prompts = []
    action_indices = []
    valids = []
    
    for text in text_actions:
        # Extract task type
        task_type, task_type_valid = extract_task_type(text)
        pred_task_types.append(task_type)
        
        # Extract task prompt
        task_prompt, task_prompt_valid = extract_task_prompt(text)
        pred_task_prompts.append(task_prompt)
        
        # Extract action
        action_index, action_valid = extract_action(text)
        action_indices.append(action_index)
        
        # All three must be valid for the output to be valid
        valids.append(1 if (task_type_valid and task_prompt_valid and action_valid) else 0)
    
    return pred_task_types, pred_task_prompts, action_indices, valids


# Legacy function for backward compatibility
def habitat_projection_legacy(text_actions: List[str], env_name):
    """
    Legacy version that only returns action_indices and valids.
    Use this if you don't need task_type prediction.
    """
    output_indices = []
    valids = []
    if env_name == 'habitat':
        action_list = ["move_forward", "turn_left", "turn_right", "look_up", "look_down", "stop"]
    else:
        raise NotImplementedError("Action list not implemented for this env!")
    for string in text_actions:
        if not isinstance(string, str):
            # directly output a random action if the string is not a string
            output_indices.append(random.randint(0, len(action_list) - 1))
            valids.append(0)
            continue
        string = string.lower()
        action_index = string.find('"action":')
        # Extract everything after "action":
        string = string[action_index:]
        contained_actions = []
        # Find all actions that are contained in the string
        for action in action_list:
            if action in string:
                contained_actions.append(action)
        # Remove duplicates by converting to a set and back to a list
        contained_actions = list(set(contained_actions))
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            # Only one keyword from action_list is in the string
            output_indices.append(action_list.index(contained_actions[0]))
            valids.append(1)
        else:
            # The string contains none or multiple keywords, randomly select an index from action_list
            output_indices.append(random.randint(0, len(action_list) - 1))
            valids.append(0)
    return output_indices, valids
