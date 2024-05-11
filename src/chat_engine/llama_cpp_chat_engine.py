from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from src.chat_engine.chat_engine import ChatEngine
from src.const import SYSTEM_PROMPT


class LlamaCPPChatEngine(ChatEngine):
    def __init__(self, model_path):
        self._model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            verbose=False
        )
        self.n_ctx = self._model.context_params.n_ctx
        self._eos_token = self._model._model.token_get_text(
            int(self._model.metadata['tokenizer.ggml.eos_token_id'])
        )
        self._formatter = Jinja2ChatFormatter(
            template=self._model.metadata['tokenizer.chat_template'],
            bos_token=self._model._model.token_get_text(
                int(self._model.metadata['tokenizer.ggml.bos_token_id'])
            ),
            eos_token=self._eos_token,
            stop_token_ids=self._model.metadata['tokenizer.ggml.eos_token_id']
        )

        self._tokenizer = self._model.tokenizer()

    def chat(self, messages, user_message, context):
        if context:
            user_message_extended = "\n".join(context + [f"Question: {user_message}"])
        else:
            user_message_extended = user_message
        messages = (
            [
                {
                    "role": "system",
                    "context": SYSTEM_PROMPT
                }
            ] + messages + [
            {
                "role": "user",
                "content": user_message_extended,

            }
        ]
        )
        prompt = self._formatter(messages=messages).prompt
        tokens = self._tokenizer.encode(prompt, add_bos=False)
        n_tokens = len(tokens)
        response_generator = self._model.create_completion(
            tokens,
            stop=self._eos_token,
            max_tokens=self.n_ctx - n_tokens,
            # max_tokens=self.n_ctx,
            # max_tokens=None,
            stream=True
        )

        return response_generator, n_tokens
