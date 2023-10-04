from typing import Optional, Iterable, Callable, Hashable, OrderedDict
import uuid
from transformers import pipeline
from collections import OrderedDict
import torch
from sentence_transformers import SentenceTransformer
import warnings


class ModelManager:
    """
    basically a wrapper around a dict[model_name, model] that handles to/from gpu automagically
    usefull in single gpu multiple model contexts
    """

    def __init__(self) -> None:
        self.models: OrderedDict[Hashable, Callable] = OrderedDict()
        self._on_gpu = None

    @property
    def on_gpu(self):
        return self._on_gpu

    @on_gpu.setter
    def on_gpu(self, new_id: Hashable):
        """
        sets self.on_gpu moves whats currently there to the cpu
        and what is being put there to the gpu

        :param new_id: model id of new self.on_gpu
        :type new_id: Hashable
        """
        if self._on_gpu == new_id:
            return
        if self._on_gpu is not None:
            self.to_device(self.models.get(self._on_gpu), "cpu")
            self.to_device(getattr(self.models.get(self._on_gpu), "tokenizer"), "cpu")

        self.to_device(self.models.get(new_id), "cuda")
        self.to_device(getattr(self.models.get(new_id), "tokenizer"), "cuda")

        self._on_gpu = new_id

    def to_device(self, mod, device="cpu"):
        if mod is None:
            return
        if device == "cpu":
            if hasattr(mod, "cpu"):
                mod.cpu()
            elif hasattr(mod, "to"):
                mod.to("cpu")
            elif hasattr(mod, "model"):
                if hasattr(mod.model, "cpu"):
                    mod.model.cpu()
                elif hasattr(mod.model, "to"):
                    mod.model.to("cpu")
        elif device == "cuda":
            if hasattr(mod, "cuda"):
                mod.cuda()
            elif hasattr(mod, "to"):
                mod.to("cuda")
            elif hasattr(mod, "model"):
                if hasattr(mod.model, "cuda"):
                    mod.model.cuda()
                elif hasattr(mod.model, "to"):
                    mod.model.to("cuda")

    def register_model(
        self, model: Callable, id: Optional[Hashable] = None
    ) -> Hashable:
        """
        register a model with the manager

        :param model: the model to use
        :type model: Callable
        :param id: model id, defaults to random uuid if None
        :type id: Optional[Hashable], optional
        :return: model id
        :rtype: Hashable
        """
        id = uuid.uuid4() if id is None else id
        self.models[id] = model
        self.to_device(self.models.get(id), "cpu")
        self.to_device(getattr(self.models.get(id), "tokenizer"), "cpu")
        return id

    def unregister_model(self, id: Hashable):
        """
        remove model with id from model registry

        :param id: model id
        :type id: Hashable
        """
        if self._on_gpu == id:
            self._on_gpu = None
        if id in self.models.keys():
            self.to_device(self.models.get(id), "cpu")
            self.to_device(getattr(self.models.pop(id), "tokenizer"), "cpu")

    def __call__(self, input):
        mod = self.models.get(self._on_gpu)
        if mod == None:
            warnings.warn("set on_gpu before calling manager")
            return
        if self._on_gpu == "encoder":
            return mod.encode(input)
        return mod(input)

    @staticmethod
    def get_default():
        """
          encoder: BAAI/bge-base-en-v1.5
            summarization: facebook/bart-large-cnn
            intent: mistralai/Mistral-7B-v0.1

        :return: model manager
        :rtype: ModelManager
        """
        mngr = ModelManager()

        encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
        mngr.register_model(encoder, "encoder")

        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
        )

        mngr.register_model(summarizer, "summarizer")

        intent = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
            
        )
        mngr.register_model(intent, "intent_generator")
        return mngr


if __name__ == "__main__":
    mngr = ModelManager.get_default()
    mngr._on_gpu
    print(mngr.models.get("encoder").device)
    print(mngr.models.get("summarizer").device)
    print(mngr.models.get("intent_generator").device)
    mngr.on_gpu = "encoder"
    print(mngr.models.get("encoder").device)
    
