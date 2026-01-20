from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str) -> ModelBase:

    if 'qwen2.5' in model_path.lower():
        from pipeline.model_utils.qwen2_5_model import Qwen2_5Model
        return Qwen2_5Model(model_path)
    elif 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    elif 'llama-3.3' in model_path.lower():
        from pipeline.model_utils.llama3_3_model import Llama3_3Model
        return Llama3_3Model(model_path)
    elif 'llama-3.1' in model_path.lower():
        from pipeline.model_utils.llama3_1_model import Llama3_1Model
        return Llama3_1Model(model_path)
    elif 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in model_path.lower():
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path) 
    elif 'yi' in model_path.lower():
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    elif 'deepseek' in model_path.lower():
        from pipeline.model_utils.deepseek_model import DeepSeek7BChatModel
        return DeepSeek7BChatModel(model_path)
    elif 'vicuna' in model_path.lower():
        from pipeline.model_utils.vicuna_model import VicunaModel
        return VicunaModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
