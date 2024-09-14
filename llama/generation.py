# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],    # prompt tokens
        max_gen_len: int,                  # generate max len
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:   # Optional[List[List[float]]]] 相当于 Union[List[List[float]], None]
                                                                # 即是一个二维浮点数列表，也可以是 None
        # Tuple不可变性
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        
        """
        根据提供的提示生成文本序列，使用语言生成模型
        
        参数：
            prompt_tokens (List[List[int]]): 经过标记化的提示列表，每个提示用一个整数列表表示。
            max_gen_len (int): 生成文本序列的最大长度。
            temperature (float, 可选): 控制采样随机性的温度值。默认值为 0.6。
            top_p (float, 可选): 用于核采样的 top-p 概率阈值。默认值为 0.9。
            logprobs (bool, 可选): 指示是否计算 token 的对数概率的标志。默认值为 False。
            echo (bool, 可选): 指示是否在生成的输出中包含 prompt token 的标志。默认值为 False。
        
        返回值：
            Tuple[List[List[int]], Optional[List[List[float]]]]: 返回一个元组，包含生成的 token 序列，如果 logprobs 为 True, 还包含相应的 token 对数概率。

        注意事项：
            此方法使用提供的 prompts 作为生成文本的基础。它采用核采样来生成具有可控随机性的文本。
            如果 logprobs 为 True, 则会计算每个生成 token 的对数概率。
        """
        
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)                 # max
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)   # 最终要生成字总长度
        
        """
        if:
            prompt_tokens = [
                [1, 2, 3],      # prompt_token 1
                [4, 5],         # prompt_token 2
                [6, 7, 8, 9]    # prompt_token 3
                ]
        
        if:
            pad_id = 0
            bsz = len(prompt_tokens)  # 批次大小为 3
            max_length = max(len(t) for t in prompt_tokens)  # 最大长度为 4
        
        初始的 tokens 张量:
            tensor([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
        
        => 填充 tokens 张量
            tensor([[1, 2, 3, 0],
                    [4, 5, 0, 0],
                    [6, 7, 8, 9]])
        """

        pad_id = self.tokenizer.pad_id                                                  # 填充字，在 tokenizer 中定义的填充字
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")  # 生成一个 shape 为 (prompt_tokens, total_len) 初始字符为 pad_id 的 tokens
        # => 填充 tokens 张量
        for k, t in enumerate(prompt_tokens):   # enumerate is useful for obtaining an indexed list （因为要 index
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
            # tokens[k, : len(t)]: 这一部分表示选择 tokens 张量的第 k 行，从第 0 列到第 len(t) - 1 列
        if logprobs:
            # 初始化对数概率张量
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
 
        prev_pos = 0                                               # 初始位置为0
        eos_reached = torch.tensor([False] * bsz, device="cuda")   # 记录是否到达结束标记
        input_text_mask = tokens != pad_id                         # mask 标记那些不是 padding 的地方
        if min_prompt_len == total_len:                            # 如果提示长度等于总长度
            logits = self.model.forward(tokens, prev_pos)          # 计算对数概率
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            # 初始时加载 prompt 部分进行预测第一个生成的 token   
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)   # 以每个句子中的[prev_pos:cur_pos]部分作为输入去推理
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)                          # 核采样
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)                                  # 再将生成的 next_token 填入 cur_pos 位置
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:                                                           # 计算对数概率
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(  # 会把当前输出的序列logits，与原始提示中的序列右移一位之后
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,                                          # ignore_index 参数的作用是忽略 target 中为 pad_id 所对应的 logits 分量
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (                     # 检查是否到达结束标记
                next_token == self.tokenizer.eos_id                               # input_text_mask 是一个布尔张量，指示哪些位置是有效的输入标记，哪些是填充标记（pad_id）
            )
            prev_pos = cur_pos
            if all(eos_reached):                                                  # 如果所有提示都到达结束标记，退出循环
                break

        """
        tokens = [
            [1, 2, 3, 0],
            [4, 5, 0, 0],
            [6, 7, 8, 9]
        ]
        
        prompt_tokens = [
            [1, 2],  # prompt_token 1
            [4],     # prompt_token 2
            [6, 7]   # prompt_token 3
        ]
        
        token_logprobs = [
            [0.1, 0.2, 0.3, 0.0],  # 对应 tokens[0]
            [0.4, 0.5, 0.0, 0.0],  # 对应 tokens[1]
            [0.6, 0.7, 0.8, 0.9]   # 对应 tokens[2]
        ]
        
        logprobs = True
        max_gen_len = 2
        echo = False
        self.tokenizer.eos_id = 0
        
    迭代处理每个提示的生成标记:
        第一轮 (i=0)
            toks = [1, 2, 3, 0]
            start = 2 , 因为 len(prompt_tokens[0]) = 2
            截取后, toks = [3, 0]（从索引 2 开始）
        第二轮 (i=1)
            toks = [4, 5, 0, 0]
            start = 1 , 因为 len(prompt_tokens[1]) = 1
            截取后, toks = [5, 0]（从索引 1 开始）
        第三轮 (i=2)
            toks = [6, 7, 8, 9]
            start = 2 , 因为 len(prompt_tokens[2]) = 2
            截取后, toks = [8, 9]（从索引 2 开始）
        
    处理对数概率:
        第一轮 (i=0)
            对应的 probs = [0.3, 0.0]（截取后）
        第二轮 (i=1)
            对应的 probs = [0.5, 0.0]（截取后）
        第三轮 (i=2)
            对应的 probs = [0.8, 0.9]（截取后）            
        
    检查结束标记并截取:
        第一轮 (i=0)
            toks = [3, 0] 包含结束标记 0
            截取后, toks = [3]
            对应 probs 变为 [0.3]
        第二轮 (i=1)
            toks = [5, 0] 包含结束标记 0
            截取后, toks = [5]
            对应 probs 变为 [0.5]
        第三轮 (i=2)
            toks = [8, 9] 不包含结束标记，保持不变
            对应 probs = [0.8, 0.9]
    
    最终结果：
        out_tokens = [[3], [5], [8, 9]]
        out_logprobs = [[0.3], [0.5], [0.8, 0.9]]
        
    out_tokens 中的每个元素都是经过处理的生成序列，仅包含有效内容。
    out_logprobs 中的对数概率对应于生成的标记，可以用于分析生成的质量。
        """
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
