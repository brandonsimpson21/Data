from typing import Optional, Iterable, Callable, Hashable, OrderedDict
import uuid
from transformers import pipeline
from collections import OrderedDict
import torch
from sentence_transformers import SentenceTransformer


class ModelManager:
    """
    basically a wrapper around a dict[model_name, model] that handles to/from gpu automagically
    usefull in single gpu multiple model contexts
    """

    def __init__(self) -> None:
        self.models: OrderedDict[Hashable, Callable] = OrderedDict()
        self._on_gpu = ""

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
        if self.models.get(self._on_gpu) is not None:
            if hasattr(self.models[self._on_gpu], "to"):
                self.models.get(self._on_gpu).to("cpu")
            else:
                if hasattr(self.models[self._on_gpu], "model"):
                    if hasattr(self.models[self._on_gpu].model, "to"):
                        self.models.get(self._on_gpu).model.to("cpu")

        if self.models.get(new_id) is not None:
            if hasattr(self.models[new_id], "to"):
                self.models[new_id].to("cuda")
            else:
                if hasattr(self.models[self.new_id], "model"):
                    if hasattr(self.models[self.new_id].model, "to"):
                        self.models.get(self.new_id).model.to("cuda")
            self._on_gpu = new_id

    def is_on_gpu(self, id: Hashable) -> bool:
        """
        is model with <id> currently on_gpu
        :param id: model id
        :type id: Hashable
        :return: true model is currently on gpu
        :rtype: bool
        """
        return True if self._on_gpu == id else False

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
        if hasattr(model, "to"):
            self.models[id] = model.to("cpu")
        else:
            self.models[id] = model
        return id

    def deregister_model(self, id: Hashable):
        """
        remove model with id from model registry

        :param id: model id
        :type id: Hashable
        """
        self._on_gpu = None
        self.models.pop(id)

    def __call__(self, input):
        return self.models.get(self._on_gpu)(input)

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
        encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        encoder = torch.compile(encoder, fullgraph=True)
        mngr.register_model(encoder, "encoder")

        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="cpu",
            torch_dtype=torch.bfloat16,
        )
        summarizer.model = torch.compile(summarizer.model, fullgraph=True)
        mngr.register_model(summarizer, "summarizer")

        intent = pipeline(
            "text-generation",
            device="cpu",
            model="mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16,
        )
        intent.model = torch.compile(intent.model, fullgraph=True)
        mngr.register_model(intent, "intent_generator")
        return mngr

    def call_sequentially(self, input, ids: Iterable[Hashable]):
        """
         apply models sequentiall by id useful for
        mod3(mod2(mod1)) style computations

        :param input: model input
        :type input: Any
        :param ids: model id
        :type ids: Iterable[Hashable]
        :return: output of sequence
        :rtype: _type_
        """
        output = input
        for id in ids:
            if id in self.models.keys():
                self.on_gpu = id
                output = self(id)
        self.on_gpu = ids[0]
        return output

    def call_iteratively(self, input, ids: Iterable[Hashable]):
        """
         apply models sequentially on input

        :param input: model input
        :type input: Any
        :param ids: model id
        :type ids: Iterable[Hashable]
        :return: output of each model
        :rtype: Iterable[Any]
        """
        output = []
        for id in ids:
            if id in self.models.keys():
                self.on_gpu = id
                output.append(self(input))
        self.on_gpu = ids[0]
        return output

    def __getitem__(self, id: Hashable) -> Callable:
        self.on_gpu = id
        return self.models.get(id)


if __name__ == "__main__":
    mngr = ModelManager()

    mock_model = lambda x: x
    mock_model2 = lambda _: -66

    id = mngr.register_model(mock_model, "mock")
    id2 = mngr.register_model(mock_model2)

    assert id == "mock"
    assert mngr.is_on_gpu(id) == False
    mngr.on_gpu = id
    assert mngr.is_on_gpu(id) == True
    assert mngr("test") == "test"
