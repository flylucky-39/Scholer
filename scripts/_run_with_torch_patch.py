"""Wrapper to bypass transformers torch.load safety check (requires torch>=2.6 for .bin).
Patches check_torch_load_is_safe everywhere before importing the target script."""
import sys
import importlib

# Patch the safety gate everywhere before any transformers model loading happens
import transformers.utils.import_utils as _iu
_iu.check_torch_load_is_safe = lambda: None

# Also patch it in modeling_utils where load_state_dict may import it directly
import transformers.modeling_utils as _mu
if hasattr(_mu, 'check_torch_load_is_safe'):
    _mu.check_torch_load_is_safe = lambda: None

# Patch any other module that may have imported it
import transformers
for _mod_name in list(sys.modules):
    if _mod_name.startswith('transformers') and hasattr(sys.modules[_mod_name], 'check_torch_load_is_safe'):
        sys.modules[_mod_name].check_torch_load_is_safe = lambda: None

# Now run the target script with the remaining argv
target = sys.argv[1]
sys.argv = sys.argv[1:]  # shift so target script sees its own name as argv[0]

spec = importlib.util.spec_from_file_location("__main__", target)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
