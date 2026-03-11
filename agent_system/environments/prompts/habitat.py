# Unified Habitat Prompt Template
# This template allows the model to infer the task type and task_prompt from the task description

HABITAT_UNIFIED_COT_TEMPLATE = """<image>You are an embodied agent in an indoor environment. Your task is to navigate and improve the score of a visual perception task.

Task: {task_description}
Current Score: {conf_score:.3f} (range 0~1)

Based on the task description, you need to:
1. Determine what type of visual perception task this is:
   - grounding: Find a specific object described in natural language
   - segment: Segment a specific object described in natural language
   - 3d-box: Predict the 3D bounding box of an object
2. Extract the target object description (task_prompt) from the task description
3. Observe the current image and decide on an action

Available actions: ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'stop']. You can only select 1 action at a time.

Your response should be a valid JSON in the following format:
{{
"thoughts": "{{First, analyze the task description to determine the task type and identify the target object. Then, describe where the target object is in the image (position, visibility, occlusion). Finally, decide which action to take.}}",
"task_type": "{{one of: grounding, segment, 3d-box}}",
"task_prompt": "{{a short phrase describing the target object, extracted from the task description}}",
"action": "{{action}}"
}}"""


# ============================================================
# Task description templates for different task types
# These are used to generate the task_description field in the unified template
# ============================================================

TASK_DESC_GROUNDING = 'Find the object: "{caption}". Navigate to improve visual grounding.'

TASK_DESC_SEGMENT = 'Segment the object: "{caption}". Navigate to improve segmentation quality.'

TASK_DESC_3D_BOX = 'Predict 3D bounding box for: "{caption}". Navigate to improve 3D-box accuracy.'

TASK_DESC_DETECT = 'The target object is outlined in red in the image. Navigate to improve detection confidence.'

TASK_DESC_CLASS_DETECT = 'Detect all objects of category "{category}". Navigate to improve detection for this class.'


# ============================================================
# Legacy templates (kept for backward compatibility)
# ============================================================

HABITAT_VISUAL_GROUNDING_COT_TEMPLATE = """<image>You are an agent in an indoor environment, and you will see the current RGB observation image. The target object is "{task_caption}. " Your goal is to improve the viusal grounding score of the target object (score range 0~1) by controlling your own movement. The current visual grounding score is "{conf_score:.3f}". You can choose between ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'stop']. You can only select 1 action at a time. Your response should be a valid json file in the following format:
{{
"thoughts": "{{First, describe the target object, such as its position and occlusion. Then, check the current observation score. Finally, think about what action to take. }}", 
"action": "{{action}}" 
}}"""

HABITAT_VISUAL_DETECT_COT_TEMPLATE    = """<image>You are an agent in an indoor environment, and you will see the current RGB observation image. The target object is outlined in red in the image. Your goal is to improve the detection score of the target object (score range 0~1) by controlling your own movement. The current detection score is "{conf_score:.3f}". You can choose between ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'stop']. You can only select 1 action at a time. Your response should be a valid json file in the following format:
{{
"thoughts": "{{First, describe the target object, such as its position and occlusion. Then, check the current observation score. Finally, think about what action to take. }}", 
"action": "{{action}}" 
}}"""

HABITAT_VISUAL_CLASS_DETECT_COT_TEMPLATE    = """<image>You are an agent in an indoor environment, and you will see the current RGB observation image. The task category is "{task_caption}. " Your goal is to improve the detector's detection score (score range 0~1) for all objects of this category within the field of view by controlling your own movement. The current detection score is "{conf_score:.3f}". You can choose between ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'stop']. You can only select 1 action at a time. Your response should be a valid json file in the following format:
{{
"thoughts": "{{First, describe the target object, such as its position and occlusion. Then, check the current observation score. Finally, think about what action to take. }}", 
"action": "{{action}}" 
}}"""

HABITAT_VISUAL_SEGMENT_COT_TEMPLATE   = """<image>You are an agent in an indoor environment, and you will see the current RGB observation image. The target object is "{task_caption}. " Your goal is to improve the segmentation score of the target object (score range 0~1) by controlling your own movement. The current segmentation score is "{conf_score:.3f}". You can choose between ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'stop']. You can only select 1 action at a time. Your response should be a valid json file in the following format:
{{
"thoughts": "{{First, describe the target object, such as its position and occlusion. Then, check the current observation score. Finally, think about what action to take. }}", 
"action": "{{action}}" 
}}"""

HABITAT_VISUAL_3D_BOX_COT_TEMPLATE   = """<image>You are an agent in an indoor environment, and you will see the current RGB observation image. The target object is "{task_caption}. " Your goal is to improve the 3D-Box prediction score of the target object (score range 0~1) by controlling your own movement. The current 3D-Box prediction score is "{conf_score:.3f}". You can choose between ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'stop']. You can only select 1 action at a time. Your response should be a valid json file in the following format:
{{
"thoughts": "{{First, describe the target object, such as its position and occlusion. Then, check the current observation score. Finally, think about what action to take. }}", 
"action": "{{action}}" 
}}"""
