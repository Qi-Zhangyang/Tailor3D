'''
This is to save and load the model.
'''

def check_model_checkpoint_consistency(ckpt_state_dict, model_state_dict, special_strs=None):
    """
    Maintain all checkpoint keys. Ignore keys with specific endings if absent. 
    Raise exception for model keys not in checkpoint unless ignored.
    ckpt: The state dictionary of the checkpoint.
    model_state_dict: The state dictionary of the model.
    special_endings: A list of specific endings of strings to be ignored.
    """
    filtered_ckpt = {}
    special_modules =[]
    for key in model_state_dict.keys():
        if key in ckpt_state_dict:
            filtered_ckpt[key] = ckpt_state_dict[key]
        elif any(special_str in key for special_str in special_strs):
            special_modules.append(key)
            continue
        else:
            raise KeyError(f"Key '{key}' not found in checkpoint and does not match any special endings.")
        
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict



# This is for reducing impact at the beginning of training.
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def filter_model_checkpoint(ckpt_state_dict, model_state_dict, need_strs=None):
    filtered_ckpt = {}
    for key in model_state_dict.keys():
        if key in ckpt_state_dict and any(need_str in key for need_str in need_strs):
            filtered_ckpt[key] = ckpt_state_dict[key]
        else:
            continue

    return filtered_ckpt
